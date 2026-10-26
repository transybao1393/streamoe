#!/usr/bin/env bash
# watch_run.sh — sample a streammoe process and the host once a second
# while it runs, so a capped run can be checked against the cap and
# against what it does to the rest of the machine.
#
#   scripts/watch_run.sh <pid> <out.csv>
#
# Columns: time, process RSS / phys_footprint (MiB), host reclaimable
# and wired (MiB), swap used (MiB), cumulative pageins / swapins,
# memory pressure level, disk read MB/s.

set -u
pid=${1:?pid}
out=${2:?out.csv}
page=$(sysctl -n hw.pagesize)

vm() { vm_stat | awk -v k="$1" -F: '$1==k {gsub(/[ .]/,"",$2); print $2}'; }

echo "t,rss_mib,footprint_mib,reclaimable_mib,wired_mib,swap_used_mib,pageins,swapins,pressure,disk_read_mbs" > "$out"
t0=$(date +%s)
while kill -0 "$pid" 2>/dev/null; do
    rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
    fp=$(footprint -p "$pid" 2>/dev/null | awk '/Footprint:/ {for (i=1;i<=NF;i++) if ($i=="Footprint:") {v=$(i+1); u=$(i+2)}; if (u ~ /GB/) v*=1024; if (u ~ /KB/) v/=1024; print int(v); exit}')
    free=$(vm "Pages free"); inact=$(vm "Pages inactive"); spec=$(vm "Pages speculative"); purg=$(vm "Pages purgeable")
    wired=$(vm "Pages wired down"); pin=$(vm "Pageins"); sin=$(vm "Swapins")
    recl=$(( (free + inact + spec + purg) * page / 1048576 ))
    wmib=$(( wired * page / 1048576 ))
    swap=$(sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if ($i=="used") {v=$(i+2); sub(/M/,"",v); print int(v)}}')
    lvl=$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null)
    rd=$(iostat -d -c 2 -w 1 disk0 2>/dev/null | tail -1 | awk '{print $3}')
    echo "$(( $(date +%s) - t0 )),$(( ${rss:-0} / 1024 )),${fp:-},$recl,$wmib,${swap:-0},$pin,$sin,${lvl:-},${rd:-}" >> "$out"
done
