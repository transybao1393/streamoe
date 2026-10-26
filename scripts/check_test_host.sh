#!/usr/bin/env bash
#
# Check that this machine can run the streammoe test tiers before
# starting a test phase: CPU, RAM, current memory headroom, load,
# disk, and whether the model files the oracle suites load exist.
#
# Usage:
#   scripts/check_test_host.sh            # report + verdict per tier
#
# Env overrides (same as the test suites):
#   STREAMMOE_SMOKE_ARTIFACTS   dir with model_weights.bin / .json, vocab.bin
#   STREAMMOE_SMOKE_ROOT        model root dir holding packed_experts/
#   MIN_FREE_GIB              headroom required for model tiers (default 8)
#
# Exit status:
#   0  unit/kernel tier and model tier both runnable
#   2  only the unit/kernel tier is runnable (run without --include-ignored)
#   1  not even the unit/kernel tier is runnable

set -u

ARTIFACTS="${STREAMMOE_SMOKE_ARTIFACTS:-/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-artifacts}"
ROOT="${STREAMMOE_SMOKE_ROOT:-/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-root}"
MIN_FREE_GIB="${MIN_FREE_GIB:-8}"

gib() { awk -v b="$1" 'BEGIN { printf "%.1f", b / 1073741824 }'; }

ok=1
fail() { echo "  FAIL  $*"; ok=0; }
pass() { echo "  ok    $*"; }
warn() { echo "  warn  $*"; }

echo "== Host"
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "  streammoe needs macOS on Apple Silicon (got $(uname -s) $(uname -m))"
    exit 1
fi
cpu="$(sysctl -n machdep.cpu.brand_string)"
ncpu="$(sysctl -n hw.ncpu)"
pcores="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo '?')"
ecores="$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo 0)"
ram="$(sysctl -n hw.memsize)"
echo "  CPU   $cpu — $ncpu cores ($pcores P + $ecores E)"
echo "  RAM   $(gib "$ram") GiB"

# Memory that can be handed to a new process without swapping:
# free + inactive + speculative + purgeable pages.
page="$(sysctl -n hw.pagesize)"
avail_pages="$(vm_stat | awk '
    /Pages free/        { gsub("\\.", "", $3); s += $3 }
    /Pages inactive/    { gsub("\\.", "", $3); s += $3 }
    /Pages speculative/ { gsub("\\.", "", $3); s += $3 }
    /Pages purgeable/   { gsub("\\.", "", $3); s += $3 }
    END { print s }')"
avail=$(( avail_pages * page ))
wired_pages="$(vm_stat | awk '/Pages wired down/ { gsub("\\.", "", $4); print $4 }')"
echo "  Avail $(gib "$avail") GiB reclaimable · wired $(gib $(( wired_pages * page ))) GiB"
load1="$(sysctl -n vm.loadavg | awk '{ print $2 }')"
echo "  Load  $load1 (1-min avg)"
disk_free="$(df -k "$PWD" | awk 'NR == 2 { print $4 * 1024 }')"
echo "  Disk  $(gib "$disk_free") GiB free on $(df -k "$PWD" | awk 'NR == 2 { print $NF }')"

echo
echo "== Tier 1: unit + kernel tests (no model files)"
if [ "$ram" -lt $(( 8 * 1073741824 )) ]; then
    fail "RAM below 8 GiB"
else
    pass "RAM ≥ 8 GiB"
fi
if [ "$avail" -lt $(( 2 * 1073741824 )) ]; then
    fail "less than 2 GiB reclaimable — close other apps first"
else
    pass "≥ 2 GiB reclaimable"
fi
if awk -v l="$load1" -v n="$ncpu" 'BEGIN { exit !(l > n) }'; then
    warn "load $load1 exceeds $ncpu cores — kernel benches will be noisy"
fi
unit_ok=$ok

echo
echo "== Tier 2: model tests (--include-ignored, a3b oracle suites)"
ok=1
weights="$ARTIFACTS/model_weights.bin"
experts="$ROOT/packed_experts"
if [ ! -f "$weights" ]; then
    fail "no $weights (set STREAMMOE_SMOKE_ARTIFACTS)"
fi
if [ ! -d "$experts" ]; then
    fail "no $experts (set STREAMMOE_SMOKE_ROOT)"
fi
if [ $ok -eq 1 ]; then
    wbytes="$(stat -f %z "$weights")"
    ebytes="$(find "$experts" -type f -name '*.bin' -exec stat -f %z {} + | awk '{ s += $1 } END { print s + 0 }')"
    echo "  model non-expert $(gib "$wbytes") GiB · experts $(gib "$ebytes") GiB"
    # Mirror io/expert_io_mode.rs: Pread when experts > 0.75 × RAM.
    if [ "$ebytes" -gt $(( ram / 4 * 3 )) ]; then
        warn "experts exceed 0.75 × RAM — the engine will pick pread mode; mmap-mode results won't apply"
    else
        pass "experts fit 0.75 × RAM — mmap mode available"
    fi
    if [ "$avail" -lt $(( MIN_FREE_GIB * 1073741824 )) ]; then
        fail "only $(gib "$avail") GiB reclaimable, need ≥ $MIN_FREE_GIB GiB (MIN_FREE_GIB)"
    else
        pass "≥ $MIN_FREE_GIB GiB reclaimable"
    fi
fi
model_ok=$ok

echo
if [ "$unit_ok" -ne 1 ]; then
    echo "Verdict: this host cannot run the test suite."
    exit 1
fi
if [ "$model_ok" -ne 1 ]; then
    echo "Verdict: unit + kernel tier only:"
    echo "  cargo test --no-fail-fast --features model-qwen3-6-35b-a3b"
    exit 2
fi
echo "Verdict: full gate runnable (model contexts contend for one GPU — keep --test-threads=1):"
echo "  cargo test --no-fail-fast --features model-qwen3-6-35b-a3b -- --include-ignored --test-threads=1 \\"
echo "    --skip resuming_prefill_after_seq_rm_matches_full_prefill"
exit 0
