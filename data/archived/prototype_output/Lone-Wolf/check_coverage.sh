#!/bin/bash

echo "=========================================="
echo "CHECKING COVERAGE OF LW-SUSPICIOUS EVENTS"
echo "=========================================="
echo ""

# LogFile LSNs to check
logfile_lsns=(
  "1548873849"
  "1548892107"
  "1551306017"
)

# UsnJrnl USNs to check (sampling key ones)
usnjrnl_usns=(
  "211756432"
  "216507152"
  "238680664"
  "238687328"
  "239459168"
  "241578752"
  "241579696"
  "241579920"
  "241584520"
  "241591696"
  "241591920"
  "241736288"
  "241737280"
  "241737456"
  "241740304"
  "241745728"
  "241745920"
  "245721000"
  "245721880"
  "245722040"
  "245725032"
  "245727232"
  "245727392"
  "245804672"
  "245805944"
  "245806152"
  "245808992"
  "245814056"
  "245814264"
  "249032136"
  "239046272"
  "239049160"
  "239049856"
  "239050536"
  "239053344"
  "239054048"
  "239054704"
  "239055432"
  "239057384"
  "239057592"
  "239059120"
  "239059816"
)

echo "=== LOGFILE LSN CHECKS ==="
found_lf=0
total_lf=${#logfile_lsns[@]}

for lsn in "${logfile_lsns[@]}"; do
  if grep -q ",$lsn\.0," flagged_files_hybrid.csv 2>/dev/null; then
    echo "✓ LogFile LSN $lsn - FOUND"
    ((found_lf++))
  else
    echo "✗ LogFile LSN $lsn - NOT FOUND"
  fi
done

echo ""
echo "LogFile Detection: $found_lf/$total_lf"
echo ""

echo "=== USNJRNL USN CHECKS ==="
found_usn=0
total_usn=${#usnjrnl_usns[@]}

for usn in "${usnjrnl_usns[@]}"; do
  if grep -q ",$usn\.0," flagged_files_hybrid.csv 2>/dev/null; then
    ((found_usn++))
  else
    echo "✗ UsnJrnl USN $usn - NOT FOUND"
  fi
done

echo ""
echo "UsnJrnl Detection: $found_usn/$total_usn"
echo ""

echo "=========================================="
echo "OVERALL COVERAGE"
echo "=========================================="
total_events=$((total_lf + total_usn))
found_events=$((found_lf + found_usn))
echo "Total suspicious events: $total_events"
echo "Events detected as HIGH: $found_events"
echo "Detection rate: $(awk "BEGIN {printf \"%.1f%%\", ($found_events/$total_events)*100}")"

