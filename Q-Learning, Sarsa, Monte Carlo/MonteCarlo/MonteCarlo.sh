#!/bin/bash

# Number of defense agents must be added by one to take into account of goalkeeper
# Cannot run an environment where defending agents exist but none are playing
# goalkeeper

epochs=5000
time_sleep=4

./../../../bin/HFO --defense-agents=2 --no-logging --offense-agents=1 --offense-on-ball 11 --trials $epochs \
        --deterministic --discrete=True --frames-per-trial 2000 --untouched-time 2000 --headless >/dev/null 2>&1 &
#--no-sync  --fullstate
sleep $time_sleep
./DiscreteHFO/Initiator.py --numTrials=$epochs  --numPlayingDefenseNPCs=1 --numAgents=1 >/dev/null 2>&1&
echo "Environment Initialized"

# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)
sleep $time_sleep

./MonteCarloBase.py --numEpisodes=$epochs --epsilon 1  --numOpponents=1  &
echo "Attacker Controller Initialized"

sleep $time_sleep
./DiscreteHFO/Goalkeeper.py --numEpisodes=$epochs >/dev/null 2>&1 &
echo "Goalkeeper Initialized"

sleep $time_sleep
./DiscreteHFO/DiscretizedDefendingPlayer.py --id=1 --numEpisodes=$epochs >/dev/null 2>&1 &
echo "Defending Player Initialized"

sleep $time_sleep
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait
