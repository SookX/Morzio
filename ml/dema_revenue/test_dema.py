from ml.dema_revenue.main import calculate_dobule_ema


def demo():
    incomes = [1000, 1200, 900, 1300, 1500, 1100]
    dema = calculate_dobule_ema(incomes)
    print("Incomes:", incomes)
    print("DEMA:", dema.tolist())


if __name__ == "__main__":
    demo()
