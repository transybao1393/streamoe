"""Type into a running `streammoe-chat` server like the page does, then
send: for `--prefill-while-typing` (docs/chat-page.md).

    python3 tools/chat/typing_check.py out.json [--port 8077] [--tokens 120]
    python3 tools/chat/typing_check.py --compare off1.json on1.json ...

Types the prompts as one conversation (three turns, so the second and
third roll back to a checkpoint, the first to an empty context): a
keystroke every 0.12-0.26 s, a pause after each word, and a `/draft`
post 300 ms after a keystroke when no other follows (the page's
debounce). Each prompt is typed with an edit part way through: the last
~15 characters are deleted (text already prefilled) and typed again.
Enter follows the last keystroke after 0.25 s. Greedy, thinking off.
Writes each turn's answer and its first-word time from Enter.

`--compare` checks that every file has the same answers (exit 1 if
not) and prints first-word times per file. Works against a server
without the flag too (its `/draft` answers ok: false), which is the
"off" baseline.
"""
import argparse, json, random, sys, threading, time, urllib.request

PROMPTS = [
    "Explain photosynthesis to a ten-year-old in one paragraph.",
    "Write a JavaScript function that checks whether a string is a palindrome.",
    "Give me a short packing list for a weekend hiking trip.",
]
DEBOUNCE = 0.3


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST")
    with urllib.request.urlopen(req) as r:
        return r.read()


def keystrokes(text, rng):
    """(gap before, box text) per keystroke: type to ~60%, delete ~15
    characters, type the rest."""
    cut = int(len(text) * 0.6)
    back = min(15, cut - 1)
    states, gaps = [], []
    def gap(ch):
        return rng.uniform(0.12, 0.26) + (rng.uniform(0.25, 0.6) if ch == " " else 0)
    for i in range(1, cut + 1):
        states.append(text[:i]); gaps.append(gap(text[i - 1]))
    gaps[-1] += 0.6  # notice the typo
    for i in range(cut - 1, cut - back - 1, -1):
        states.append(text[:i]); gaps.append(rng.uniform(0.08, 0.12))
    gaps[-1] += 0.4
    for i in range(cut - back + 1, len(text) + 1):
        states.append(text[:i]); gaps.append(gap(text[i - 1]))
    return list(zip(gaps, states))


def type_and_send(base, prompt, tokens, rng, drafts):
    """Type `prompt`, press Enter; returns (answer, first-word s from Enter, done event)."""
    keys = keystrokes(prompt, rng)
    threads = []
    for i, (_, box) in enumerate(keys):
        nxt = keys[i + 1][0] if i + 1 < len(keys) else 0.25
        if nxt > DEBOUNCE:
            time.sleep(DEBOUNCE)
            # The page posts without waiting for the answer.
            t = threading.Thread(target=post, args=(base, "/draft", {"message": box.strip(), "system": ""}))
            t.start(); threads.append(t)
            drafts.append(box.strip())
            time.sleep(nxt - DEBOUNCE)
        else:
            time.sleep(nxt)
    body = {"message": prompt, "think": False, "temperature": 0, "max_tokens": tokens,
            "top_p": 0.8, "top_k": 20, "system": "", "image": None}
    t0 = time.time()
    req = urllib.request.Request(base + "/chat", data=json.dumps(body).encode(), method="POST")
    text, ev, first, prefill = "", None, None, None
    with urllib.request.urlopen(req) as r:
        for raw in r:
            line = raw.decode().rstrip("\n")
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: "):
                d = json.loads(line[6:])
                if ev == "done":
                    for t in threads:
                        t.join()
                    return text, first, d, prefill
                if ev == "error":
                    raise RuntimeError(d["error"])
                if ev == "prefill":
                    prefill = d
                if ev is None:
                    if first is None:
                        first = time.time() - t0
                    text += d["t"]
            elif line == "":
                ev = None
    raise RuntimeError("stream ended without done")


def run(a):
    base = f"http://127.0.0.1:{a.port}"
    rng = random.Random(a.seed)
    post(base, "/reset", {})
    info = json.loads(urllib.request.urlopen(base + "/info").read())["info"]
    res = {"prefill_while_typing": info.get("prefill_while_typing", False), "turns": []}
    for p in PROMPTS:
        drafts = []
        text, first, done, prefill = type_and_send(base, p, a.tokens, rng, drafts)
        r = {"prompt": p, "text": text, "first_word_s": first, "tok_s": done["tok_s"],
             "tokens": done["tokens"], "prompt_tokens": done["prefill_tokens"],
             "draft_tokens": done.get("draft_tokens"), "prefill_secs": prefill and prefill["secs"],
             "drafts": len(drafts), "hit": done["hot"] and done["hot"]["hit_rate"],
             "peak_gib": done["memory"]["peak_gib"]}
        res["turns"].append(r)
        print(f"  first word {first:.2f} s  prompt {r['prompt_tokens']} tok ({r['draft_tokens']} while typing,"
              f" {r['drafts']} drafts)  {r['tokens']} tok at {r['tok_s']:.1f} tok/s  peak {r['peak_gib']:.2f} GiB",
              flush=True)
        time.sleep(1.0)  # reading the answer
    json.dump(res, open(a.out, "w"), indent=1)


def compare(files):
    runs = [(f, json.load(open(f))) for f in files]
    ref = [t["text"] for t in runs[0][1]["turns"]]
    ok = True
    for f, r in runs:
        same = [t["text"] == x for t, x in zip(r["turns"], ref)]
        ok &= all(same)
        fw = " ".join(f"{t['first_word_s']:.2f}" for t in r["turns"])
        tag = "on " if r["prefill_while_typing"] else "off"
        print(f"{tag} {f}: first word {fw} s  same answers: {same}")
    print("all answers identical" if ok else "ANSWERS DIFFER")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--compare", nargs="+")
    a = ap.parse_args()
    if a.compare:
        sys.exit(0 if compare(a.compare) else 1)
    run(a)
