"""Check a `streammoe-chat --max-chats N` server: answers decoded together
must equal the same answers decoded one at a time.

    python3 tools/chat/multi_chat_check.py [--port 8077] [--chats N] [--tokens 120]

Sends the prompts one after another (each a new conversation), then N
of them at once, all greedy, and compares the texts. Prints total tok/s
(every chat's tokens over the wall time, prefills included) for both
and each chat's decode tok/s. Exit status 1 if any text differs.
"""
import argparse, json, sys, threading, time, urllib.request

PROMPTS = [
    "Explain photosynthesis to a ten-year-old in one paragraph.",
    "Write a JavaScript function that checks whether a string is a palindrome.",
    "Give me a short packing list for a weekend hiking trip.",
    "What are the main differences between TCP and UDP? Answer in a few sentences.",
]


def turn(base, prompt, tokens):
    """One greedy turn as a new conversation (no session id). Returns (text, done event)."""
    body = json.dumps({"message": prompt, "think": False, "temperature": 0, "max_tokens": tokens,
                       "system": "", "image": None}).encode()
    text, ev = "", None
    with urllib.request.urlopen(urllib.request.Request(base + "/chat", data=body, method="POST")) as r:
        for raw in r:
            line = raw.decode().rstrip("\n")
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: "):
                d = json.loads(line[6:])
                if ev == "done":
                    return text, d
                if ev == "error":
                    raise RuntimeError(d["error"])
                if ev is None:
                    text += d["t"]
            elif line == "":
                ev = None
    raise RuntimeError("stream ended without done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--chats", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=120)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    info = json.load(urllib.request.urlopen(base + "/info"))["info"]
    n = min(a.chats, info.get("max_chats", 1), len(PROMPTS))
    if n < 2:
        sys.exit(f"the server serves {info.get('max_chats', 1)} chat(s) at once; start it with --max-chats 2..4")
    prompts = PROMPTS[:n]

    t0 = time.time()
    alone = [turn(base, p, a.tokens) for p in prompts]
    wall_alone = time.time() - t0

    together = [None] * n
    def run(i):
        together[i] = turn(base, prompts[i], a.tokens)
    th = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in th: t.start()
    for t in th: t.join()
    wall_together = time.time() - t0

    toks = lambda rs: sum(d["tokens"] for _, d in rs)
    print(f"{n} chats, {info['precision']}-bit experts, cap {info['cap_gib']} GiB")
    print(f"  one at a time  {toks(alone) / wall_alone:6.2f} tok/s total")
    print(f"  together       {toks(together) / wall_together:6.2f} tok/s total")
    bad = 0
    for p, (ta, da), (tt, dt) in zip(prompts, alone, together):
        same = ta == tt
        bad += not same
        print(f"  {'identical' if same else 'DIFFERS  '}  {da['tokens']:3d} tok  alone {da['tok_s']:5.2f} tok/s  "
              f"together {dt['tok_s']:5.2f} tok/s ({dt.get('chats', 1):.1f} chats)  {p[:40]}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
