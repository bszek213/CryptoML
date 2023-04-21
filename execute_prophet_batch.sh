#!/bin/bash

list_crypt=("bitcoin" "ethereum" "cardano" "polygon" "doge" "solana" "polkadot" "shibainu" "tron" "filecoin" "chainlink" "apecoin" "mana" "avalanche" "zcash" "internetcomputer" "flow" "elrond" "tezos")

for i in "${list_crypt[@]}"
do
   python forecst_sentiment.py all "$i"
done
