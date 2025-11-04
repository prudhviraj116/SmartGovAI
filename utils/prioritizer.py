
def compute_urgency(risk_score, predicted_increase, resource_availability, weights=(0.5,0.3,0.2)):
    alpha, beta, gamma = weights
    urgency = alpha * float(risk_score) + beta * float(predicted_increase) + gamma * (1 - float(resource_availability))
    if urgency < 0: urgency = 0
    if urgency > 1: urgency = 1
    return round(urgency, 3)
