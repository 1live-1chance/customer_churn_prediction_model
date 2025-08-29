from model import ModelTrainer


def main():
    trainer = ModelTrainer(
        path='./data/dataset.csv',
        target_column='Churn'  
    )

    trainer.complete_analysis()

    results = trainer.train(epochs=50, batch_size=32)
    trainer.plot_predictions_analysis('analysis_results/predictions_analysis.png')


if __name__ == "__main__":
    main()