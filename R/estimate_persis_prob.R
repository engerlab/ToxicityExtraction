# Load the required package
library(interval)

# Read the data from the text file
data <- read.table("../path to the txt file containing the intervals", header = TRUE)

# Display the data
print(data)

# Fit the Turnbull estimator
fit <- icfit(data$l, data$u)

# View the survival probabilities
summary(fit)

# Plot the time to event function
plot(
  fit, 
  main = "Time-to-event (resoltion of toxicty) function", 
  xlab.custom = "Time", 
  ylab.custom = "Persistence Probability"
)

