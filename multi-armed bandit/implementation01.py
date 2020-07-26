"""
In this example, we play a gambling machine. There are T total machines
in the room, each of them containing different win ratios.
We know that one will provide us the best outcomes, but we will
use AI to determine which machine that is. 

For this AI method, we'll be using the Thompson Sampling Model.
More can be found here: https://stanford.io/2ZGMlnP
We want to find the machine that provides the most wins
over a specific time period.

If you'd like to change some parameters to see different
setups run, scroll down to the 'SETUP' section.
"""
import numpy as np

# To circumvent random occasions of printing in scientific notation for integer values
np.set_printoptions(suppress=True)


######### SETUP HERE #########
totalMachines_T = 5
maxWinRatio = .2
samples_N = 10000
######### END SETUP #########

# Quick Parameter Formatting
try:
    totalMachines_T = abs(int(totalMachines_T))
    maxWinRatio = abs(maxWinRatio)
    samples_N = abs(int(samples_N))
except:
    print("\n\n\tInvalid Parameters! Please check your setup values.\n\n")
    exit()

# Generate random win ratios for T total machines
machineWinRates = np.zeros(totalMachines_T)
for machine in range(totalMachines_T):
    while machineWinRates[machine] == 0:
        winRate = np.random.rand()
        # We don't want too good of a machine and a machine that doesn't win at all
        if winRate <= maxWinRatio and winRate > 0:
            machineWinRates[machine] = winRate

# Create the environment for the AI
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

# This section is where the machine begins Thompson Sampling
for sample in range(samples_N):
    chosenMachine = 0
    maxGuess = 0
    for machine in range(numMachines_d):
        # This is where a Beta distribution compares loses and wins over rounds played
        # and determines out of the following options which machine it should pick
        randomChoice = np.random.beta(playWins[machine] + 1, playLoses[machine] + 1)
        if randomChoice > maxGuess:
            maxGuess = randomChoice
            chosenMachine = machine
    # Rewarding the AI with wins (1) or provide negative feedback (0)
    if sampleOutput[sample][chosenMachine] == 1:
        playWins[chosenMachine] += 1
    else:
        playLoses[chosenMachine] += 1
sampleDistributions = playLoses + playWins

# Display results of sample runs to user in an overly excessive formatted view
print("\n\n\t\tData results of this run")
print("=" * 55 + "\n")
bestMachine = max(sampleDistributions)
bestMachineIndex = np.argmax(sampleDistributions) + 1
for machine in range(len(machineWinRates)):
    print(f"\t\tMachine {machine + 1} Win Ratio: {machineWinRates[machine]:.2f}")
    print("\t\t" + "-" * 28)
    print(f"\t\t  Wins: {int(playWins[machine])}    Loses: {int(playLoses[machine])}\n")
print("=" * 55)
print(f"\n\tThe best machine to play was Machine {bestMachineIndex}, with \n\t{int(bestMachine)} total choice distributions of {samples_N}.\n\n")