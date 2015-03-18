from scipy.stats import beta

def beta_pdf(a, b, values):
    pdf = beta.pdf(values, a, b)
    return pdf.tolist()
