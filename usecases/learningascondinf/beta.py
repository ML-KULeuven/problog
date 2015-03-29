from scipy.stats import beta

def beta_pdf(a, b, values):
    probs = []
    start = 0

    for idx in range(len(values)-1):
        end = values[idx] + (values[idx+1] - values[idx])/2
        w = end - start
        pdf = beta.pdf(values[idx],a,b).item()
        probs.append(pdf*w)
        start = end

    w = 1.0 - start
    pdf = beta.pdf(values[-1],a,b).item()
    probs.append(pdf*w)

    return probs

