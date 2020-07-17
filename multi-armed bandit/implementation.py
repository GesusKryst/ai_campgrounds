"""
In this example, we play a gambling machine. There are five machines
in the room, each of them containing different win ratios.
We know that one will provide us the best outcomes, but we will
use AI to determine which machine that is. 

For this AI method, we'll be using the Thompson Sampling Model.
More can be found here: https://stanford.io/2ZGMlnP
We want to find the machine that provides the most wins
over a specific time period.
"""
import numpy as np


# Create the environment for the AI
    # Implement random assortment of win rates and total machines
    # For example, machine 1 provides a 15/100 win ratio, machine 2 provides 4/100, ...
machineWinRates = [0.15, 0.04, 0.13, 0.11, 0.05] 
samples_N = 10000
numMachines_d = len(machineWinRates)

# Create data for every sample
sampleOutput = np.zeros((samples_N, numMachines_d))
for sample in range(samples_N):
    for machine in range(numMachines_d):
        # 
        if np.random.rand() < machineWinRates[machine]:
            sampleOutput[sample][machine] = 1 # Meaning playing at that machine at that point in time won

# Summarizing the total wins and losses per machine
playWins = np.zeros(numMachines_d) # Tracks every time a machine won
playLoses = np.zeros(numMachines_d) # Tracks every time a machine lost
for sample in sampleOutput:
    for machine in range(numMachines_d):
        if sample[machine] == 1:
            playWins[machine] += 1
        else:
            playLoses[machine] += 1

# Since we know machine 1 has the highest win rate, we can expect the total wins to be higher
print(f"Wins: {playWins}\nLoses: {playLoses}")

for sample in range(samples_N):
    chosenMachine = 0
    bestGuess = 0