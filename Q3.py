import numpy as np

# Set the random seed to reproduce results
np.random.seed(420)

# Set the possible move of bee
positions = np.array([[0, 1], [0, -1], [np.sqrt(2), np.sqrt(2)], [np.sqrt(2), -np.sqrt(2)], [-np.sqrt(2), np.sqrt(2)], [-np.sqrt(2), -np.sqrt(2)]])

# Set the equal probabilities
probabilities = [1/6 for x in range(len(positions))]

# Number of moves
moves_13 = 13

# Initiate the numpy array to store the average distance for each experiment, simulate for 100000 times
results_13 = np.empty(100000)

# Use statistical simulation to model 100000 times, 13 moves for each time, and calculate the expected distance
for i in range(len(results_13)):
    start_13 = [0, 0]
    end_13 = [0, 0]
    
    outcome = np.random.choice(len(positions), p=probabilities, size=moves_13)
    
    for j in positions[outcome]:
        end_13[0] += j[0]
        end_13[1] += j[1]
    
    distance  =  np.sqrt((end_13[1] - start_13[1])**2 + (end_13[0] - start_13[0])**2)
    results_13[i] = distance
    
    avg_distance_13 = np.mean(results_13)
    std_distance_13 = np.std(results_13)

# After T=13 steps, what is the expected value of the bee's distance from the starting hexagon?
print("After 13 steps, the expected value of the bee's distance from the starting hexagon is {:.5f}".format(avg_distance_13))

#A fter T=13 steps, what is the standard deviation of the bee's distance from the starting hexagon?
print("After 13 steps, the standard deviation of the bee's distance from the starting hexagon is {:.5f}".format(std_distance_13))

# After T=13 steps, what is the expected value of the bee's distance from the starting hexagon if we know the bee is at least 4 units away?
results_updates = results_13[results_13 >= 4]
new_distance = np.mean(results_updates)
print("After 13 steps, the expected value of the bee's distance from the starting hexagon is {:.5f} if the bee is at least 4 units away".format(new_distance))


# Use statistical simulation to model 100000 times, 60 moves for each time, and calculate the expected distance
moves_60 = 60

start_60 = [0, 0]
end_60 = [0, 0]

results_60 = np.empty(100000)

for i in range(len(results_60)):
    start_60 = [0, 0]
    end_60 = [0, 0]
    
    outcome = np.random.choice(len(positions), p=probabilities, size=moves_60)
    
    for j in positions[outcome]:
        end_60[0] += j[0]
        end_60[1] += j[1]
    
    distance  =  np.sqrt((end_60[1] - start_60[1])**2 + (end_60[0] - start_60[0])**2)
    results_60[i] = distance
    
    avg_distance_60 = np.mean(results_60)
    std_distance_60 = np.std(results_60)

# After T=60 steps, what is the expected value of the bee's distance from the starting hexagon?
print("After 60 steps, the expected value of the bee's distance from the starting hexagon is {:.5f}".format(avg_distance_60))

# After T=60 steps, what is the standard deviation of the bee's distance from the starting hexagon?
print("After 60 steps, the standard deviation of the bee's distance from the starting hexagon is {:.5f}".format(std_distance_60))

# After T=60 moves, what is the probability that the bee is at least 20 units away from the starting hexagon, given it is at least 15 units away?
results_updates = results_60[results_60 >= 15]
results_probability = np.mean(results_updates > 20)
print("the probability that the bee is at least 20 units away from the starting hexagon, given it is at least 15 units away is {:.5f}".format(results_probability))


