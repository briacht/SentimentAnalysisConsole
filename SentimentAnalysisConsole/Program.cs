using Microsoft.ML;
using System;

namespace SentimentAnalysisConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            // 1. Create new ML.NET Environment
            var mlContext = new MLContext();

            // 2. Load data
            IDataView trainingData = mlContext.Data.LoadFromTextFile<ModelInput>("yelp_train.txt", hasHeader:true, separatorChar:'\t');

            // 3. Transform data
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(ModelInput.Comment));

            // 4. Add algorithm
            var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Sentiment", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            Console.WriteLine("Training model...");
            // 5. Train model
            ITransformer trainedModel = trainingPipeline.Fit(trainingData);

            Console.WriteLine("Evaluating model...");
            // 6. Evaluate model
            IDataView testData = mlContext.Data.LoadFromTextFile<ModelInput>("yelp_test.txt", hasHeader: true, separatorChar: '\t');

            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Sentiment", scoreColumnName: "Score");
            Console.WriteLine($"\tAccuracy: {metrics.Accuracy}");

            // 7. Save trained model
            Console.WriteLine("Saving model...");
            mlContext.Model.Save(trainedModel, trainingData.Schema, "../../../model.zip");

            // 8. Use model to make prediction
            ModelInput input = new ModelInput { Comment = "I love this movie!" };

            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);
            var result = predEngine.Predict(input);

            string x = (result.Prediction == true) ? "Negative" : "Positive";

            Console.WriteLine($"Using model to predict sentiment...\n\tComment: '{input.Comment}'\n\tPredicted Sentiment: {x}");
        }
    }
}
