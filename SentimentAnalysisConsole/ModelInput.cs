using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace SentimentAnalysisConsole
{
    public class ModelInput
    {
        [LoadColumn(0)]
        public string Comment { get; set; }
        
        [LoadColumn(1)]
        public bool Sentiment { get; set; }
    }
}
