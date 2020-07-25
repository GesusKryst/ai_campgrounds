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

# To circumvents random occasions of printing in scientific notation for integer values
np.set_printoptions(suppress=True)

# Generate random win ratios for T total machines
# OG: machineWinRates = [0.15, 0.04, 0.13, 0.11, 0.05] 
total_T = 5
maxWinRatio = 0.2 
machineWinRates = np.zeros(total_T)
for machine in range(total_T):
    while machineWinRates[machine] == 0:
        winRate = np.random.rand()
        # We don't want too good of a machine and a machine that doesn't win at all
        if winRate <= maxWinRatio and winRate > 0:
            machineWinRates[machine] = winRate

# Create the environment for the AI
samples_N = 10000
numMachines_d = len(machineWinRates)

# Create data for every sample
sampleOutput = np.zeros((samples_N, numMachines_d))
for sample in range(samples_N):
    for machine in range(numMachines_d):
        # randVal is a value chosen at random between 0 and 1.
        randVal = np.random.rand()
        if randVal < machineWinRates[machine]:
            sampleOutput[sample][machine] = 1 # Meaning playing at that machine at that point in time won / randVal is lower or equal to the machine win rate

# Summarizing the total wins and losses per machine
playWins = np.zeros(numMachines_d) # Tracks every time a machine won
playLoses = np.zeros(numMachines_d) # Tracks every time a machine lost


# Since we know machine 1 has the highest win rate, we can expect the total wins to be higher
# This section is where the machine begins Thompson Sampling
for sample in range(samples_N):
    chosenMachine = 0
    maxGuess = 0
    for machine in range(numMachines_d):
        randomChoice = np.random.beta(playWins[machine] + 1, playLoses[machine] + 1)
        if randomChoice > maxGuess:
            maxGuess = randomChoice
            chosenMachine = machine
    if sampleOutput[sample][chosenMachine] == 1:
        playWins[chosenMachine] += 1
    else:
        playLoses[chosenMachine] += 1

print("Conversion rates for this run:")
for machine in range(len(machineWinRates)):
    print(f"\tMachine {machine + 1}: {machineWinRates[machine]:.2f}")
print(f"Wins: {playWins}\nLoses: {playLoses}")