from .analysis import CZSCAnalysis


def main():
    analyzer = CZSCAnalysis()
    analyzer.monitor_stocks(interval=300)


if __name__ == "__main__":
    main()
