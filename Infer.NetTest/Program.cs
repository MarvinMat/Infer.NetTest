using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;

namespace ProcessSimulator
{
    public enum Shift
    {
        Day,
        Night
    }

    public class InferenceModel
    {
        private readonly Variable<double> temperatureVar;
        private readonly Variable<bool> workingDayVar;
        private readonly Variable<double> daysSinceLastInterruptVar;
        private readonly Variable<Shift> shiftVar;
        private readonly Variable<double> operationDurationVar;

        private readonly InferenceEngine engine;

        public InferenceModel()
        {
            //temperatureVar = Variable.New<double>().Named("Temperature");
            //workingDayVar = Variable.New<bool>().Named("Working Day");
            daysSinceLastInterruptVar = Variable.New<double>().Named("Days Since Last Interrupt");
            shiftVar = Variable.New<Shift>().Named("Shift");

            operationDurationVar = Variable.GaussianFromMeanAndVariance(1, 0.01).Named("Initial Operation duration factor");

            //operationDurationVar *= (0.9 + 0.2 * (Variable.Exp(Variable.Min(Variable.Constant(30.0), daysSinceLastInterruptVar)) / Math.Exp(30))).Named("Time Since Last Interrupt Factor");
            
            //operationDurationVar *= (temperatureVar / 20).Named("Temperature Factor");

            var nightShiftWeekendFactor = Variable.New<double>().Named("Night Shift Weekend Factor");

            var isNightShiftVar = (shiftVar == Shift.Night).Named("Is Night Shift");
            using (Variable.If(isNightShiftVar))
            {
                nightShiftWeekendFactor.SetTo(Variable.Constant(1.1));
            }
            using (Variable.IfNot(isNightShiftVar))
            {
                nightShiftWeekendFactor.SetTo(Variable.Constant(1.0));
            }
            operationDurationVar *= nightShiftWeekendFactor;
            operationDurationVar.Named("Operation duration factor");

            engine = new InferenceEngine();
            // Set the compiler choice to Roslyn explicitly
            engine.Compiler.CompilerChoice = CompilerChoice.Roslyn;

            // Configure other properties if necessary
            engine.Compiler.GenerateInMemory = true; // Keep assemblies in memory if not saving to disk
            engine.Compiler.WriteSourceFiles = true;
            engine.Compiler.GeneratedSourceFolder = @"C:\Users\marvi\source\repos\Infer.NetTest\Infer.NetTest"; // Specify the folder to save graphs
            engine.SaveFactorGraphToFolder = @"C:\Users\marvi\source\repos\Infer.NetTest\Infer.NetTest";
        }

        public double Infer(double daysSinceLastInterrupt, Shift shift)
        {
            daysSinceLastInterruptVar.ObservedValue = daysSinceLastInterrupt;
            shiftVar.ObservedValue = shift;

            var operationDurationDistribution = engine.Infer(operationDurationVar);
            if (operationDurationDistribution is not Gaussian operationDurationGaussian)
                return 1;  // Fallback in case of non-Gaussian result

            return operationDurationGaussian.Sample();  // Returns a sample from the Gaussian distribution
        }

        static void Main(string[] args)
        {
            InferenceModel model = new InferenceModel();
            double inferredDuration = model.Infer(10.0, Shift.Day);
            System.Console.WriteLine($"Inferred Operation Duration: {inferredDuration}");
        }
    }
}
