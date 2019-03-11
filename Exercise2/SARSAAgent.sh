#!/bin/bash

# Number of defense agents must be added by one to take into account of goalkeeper
# Cannot run an environment where defending agents exist but none are playing
# goalkeeper
SleepTime=3
epochs=500

./../../../bin/HFO --defense-agents=2 --offense-agents=1 --offense-on-ball 11 --trials $epochs  --headless --deterministic --discrete=True --frames-per-trial 2000 --untouched-time 2000 &
sleep $SleepTime
./DiscreteHFO/Initiator.py --numTrials=$epochs --numPlayingDefenseNPCs=1 --numAgents=1 &
echo "Environment Initialized"
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)

sleep $SleepTime
./SARSABase.py --numOpponents=1 --numEpisodes=$epochs &
echo "Attacker Controller Initialized"

sleep $SleepTime
./DiscreteHFO/Goalkeeper.py --numEpisodes=$epochs &
echo "Goalkeeper Initialized"

sleep $SleepTime
./DiscreteHFO/DiscretizedDefendingPlayer.py --numEpisodes=$epochs --id=1 &
echo "Defending Player Initialized"

sleep $SleepTime
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait
