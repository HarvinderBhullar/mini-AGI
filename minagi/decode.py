"""Choosing the next character.

Nothing here draws from a random number generator. `pick_next` at temperature
zero takes the highest scoring character, and holds off degeneration with an
adaptation trace - what was just said is suppressed, and the suppression
decays - the way a neuron that has just fired is briefly harder to fire again.
`contrastive_generate` scores candidates against the context already in the
cache (Su & Collier 2022).

Variety comes from the state the model is in - which experts are loaded, how
many latent passes it took, what it has just read - not from sampling noise.
"""


import torch
import torch.nn.functional as F

from .precision import amp


def pick_next(logits, prev_ids, temperature=0.0, top_k=0, top_p=1.0,
              rep_penalty=1.0, no_repeat_ngram=0,
              adapt_strength=0.0, adapt_decay=0.88, adapt_window=64):
    """
    Choose the next token. Deterministic when temperature <= 0.

    Determinism is worth having: a model you can test and diff needs to give
    the same answer twice. But greedy decoding is also the classic cause of
    degenerate repetition - argmax on a distribution whose mode reinforces
    itself will re-enter the same state forever. Extra capacity does not fix
    that, because the loop lives in the decoding rule, not the parameters.

    Three knobs suppress loops without giving up determinism:

      adapt_strength   ADAPTATION, and the one that works. A decaying trace
                       per character: what was just said is suppressed, and
                       the suppression fades. Measured over six prompts, mean
                       repeated 8-grams:

                           greedy alone                       73.5%
                           greedy + rep_penalty 1.12          59.0%
                           greedy + adaptation 2.5            11.6%

      rep_penalty      divides the logits of tokens already emitted. Weak,
                       because it marks a character FOREVER the first time it
                       appears - after 140 characters that is 93% of the
                       alphabet, penalised equally, which is no signal. Not
                       useless (59% against 73.5% without it), but adaptation
                       does the same job properly and they were not measured
                       together.

      no_repeat_ngram  hard-bans any n-gram that has already occurred. Keep it
                       large (12+) at character level, or better, leave it off:
                       100% of real code files repeat a 32-gram and 40% repeat
                       a 48-gram, so any value strict enough to stop a loop
                       also forbids ordinary code. Measured, it barely helped -
                       the loop simply steps around the ban and resumes.

    WHY ADAPTATION AND NOT A LONGER MEMORY. The trace is deliberately short:
    at decay 0.88 its half-life is 5.4 characters, so one use of a character
    costs 2.5 logits now and 0.9 eight characters later - invisible to ordinary
    text. A character stuck in a loop never gets to decay, and converges to
    strength/(1-decay) = 20.8 logits, which no loop survives.

    Its blind spot is a loop whose period exceeds that memory. Nothing here
    catches a phrase that returns every forty characters; only the model
    getting better does.

    A frequency-normalised variant - charging a character only for use above
    its natural rate, so space is not billed 4.3 logits for being 20% of all
    text - was measured and is WORSE, 42.5% against 11.6%. Loops are built out
    of common characters, so exempting them is what lets the loop through.
    """
    logits = logits.clone()
    B = logits.shape[0]
    if adapt_strength and prev_ids is not None and prev_ids.numel():
        # The trace is a function of the history, not state carried between
        # calls, so this stays pure and every existing caller keeps working.
        # Only the last `adapt_window` characters matter: at decay 0.88,
        # 0.88^64 is 3e-4, and everything older is already nothing.
        n = min(prev_ids.shape[1], adapt_window)
        tail = prev_ids[:, -n:]
        w = (adapt_decay ** torch.arange(n - 1, -1, -1, device=logits.device,
                                         dtype=logits.dtype))
        trace = torch.zeros_like(logits)
        trace.scatter_add_(1, tail, w.expand(B, n))
        logits = logits - adapt_strength * trace
    if rep_penalty and rep_penalty != 1.0 and prev_ids is not None:
        for b in range(B):
            uniq = torch.unique(prev_ids[b])
            l = logits[b, uniq]
            logits[b, uniq] = torch.where(l > 0, l / rep_penalty,
                                          l * rep_penalty)
    if no_repeat_ngram and prev_ids is not None:
        n = no_repeat_ngram
        for b in range(B):
            seq = prev_ids[b].tolist()
            if len(seq) >= n:
                prefix = tuple(seq[-(n - 1):])
                banned = {seq[i + n - 1] for i in range(len(seq) - n + 1)
                          if tuple(seq[i:i + n - 1]) == prefix}
                if banned:
                    logits[b, list(banned)] = float("-inf")
    if temperature is None or temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    z = logits / temperature
    if top_k:
        kth = torch.topk(z, min(top_k, z.size(-1)))[0][..., -1:]
        z = z.masked_fill(z < kth, float("-inf"))
    probs = F.softmax(z, dim=-1)
    if top_p and top_p < 1.0:
        srt, order = torch.sort(probs, descending=True, dim=-1)
        cums = srt.cumsum(-1)
        srt[cums - srt > top_p] = 0.0
        srt = srt / srt.sum(-1, keepdim=True)
        probs = torch.zeros_like(probs).scatter(-1, order, srt)
    return torch.multinomial(probs, 1)




@torch.no_grad()
def contrastive_generate(model, idx, max_new_tokens, top_k=8, alpha=0.6,
                         stop_ids=None):
    """
    Contrastive search: deterministic, and anti-degeneration by construction.

        score(v) = (1 - alpha) * p(v)  -  alpha * max_j cos(h_v, h_j)

    The first term is the model's own confidence; the second penalises a
    candidate whose hidden representation looks like something already in the
    context. Loops are suppressed because a looping token is, by definition,
    representationally close to what came before.

    This matters because greedy degeneration is a symptom of maximum-likelihood
    training - MLE piles probability onto locally-safe continuations, so argmax
    walks into a cycle. Temperature does not fix that; it injects noise so the
    walk falls out of the cycle by luck. This addresses the mechanism instead,
    and the output stays a deterministic function of the input.

    Cost: one extra batched forward step of width top_k per token, sharing the
    context's KV cache. (Su & Collier 2022)
    """
    model.eval()
    device = idx.device
    n_layer = model.cfg.n_layer
    caches = [{"k": None, "v": None} for _ in range(n_layer)]
    with amp(idx.device):
        logits, _, hidden = model(idx, caches=caches, pos_offset=0,
                                  return_hidden=True)
    ctx = F.normalize(hidden[0].float(), dim=-1)          # [T, D]
    offset = idx.shape[1]
    out = idx

    for _ in range(max_new_tokens):
        probs = F.softmax(logits[0, -1].float(), dim=-1)
        top_p_vals, top_i = probs.topk(min(top_k, probs.numel()))
        k = top_i.numel()
        # replay the same context for every candidate in one batched step
        exp = [{"k": c["k"].repeat(k, 1, 1, 1),
                "v": c["v"].repeat(k, 1, 1, 1)} for c in caches]
        cand = top_i.view(k, 1)
        with amp(cand.device):
            lg, _, h = model(cand, caches=exp, pos_offset=offset,
                             return_hidden=True)
        hv = F.normalize(h[:, -1].float(), dim=-1)        # [k, D]
        penalty = (hv @ ctx.T).max(dim=-1).values         # [k]
        score = (1 - alpha) * top_p_vals - alpha * penalty
        win = int(score.argmax())

        caches = [{"k": c["k"][win:win + 1], "v": c["v"][win:win + 1]}
                  for c in exp]
        nxt = top_i[win].view(1, 1)
        out = torch.cat([out, nxt], dim=1)
        ctx = torch.cat([ctx, hv[win:win + 1]], dim=0)
        logits = lg[win:win + 1]
        offset += 1
        if stop_ids is not None and int(nxt.item()) in stop_ids:
            break
        if out.shape[1] >= model.cfg.block:
            break
    return out


