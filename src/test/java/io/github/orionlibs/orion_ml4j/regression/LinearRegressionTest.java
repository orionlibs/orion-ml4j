package io.github.orionlibs.orion_ml4j.regression;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.orionlibs.orion_ml4j.ATest;
import io.github.orionlibs.orion_runnable.functions.OrionFunction1x1;
import io.github.orionlibs.orion_simple_math.probability.distribution.NormalDistribution;
import io.github.orionlibs.orion_simple_math.random.RandomNumberGenerationService;
import io.github.orionlibs.orion_tuple.Pair;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.TestInstance.Lifecycle;

@TestInstance(Lifecycle.PER_CLASS)
//@Execution(ExecutionMode.CONCURRENT)
public class LinearRegressionTest extends ATest
{
    @Test
    void test_LinearRegression()
    {
        List<Pair<Float, Float>> pairsOfInputsOutputsTrainingData = new ArrayList<>();
        float[] inputs = new float[100];
        float[] outputs = new float[inputs.length];
        float slope = 3.0f;
        float intercept = 2.0f;
        float modelError = NormalDistribution.getRandomNumber();
        //Y = 2 + 3X
        OrionFunction1x1<Float, Float> modelFunction = (Float x) -> intercept + (slope * x) + modelError;
        for(int i = 0; i < inputs.length; i++)
        {
            inputs[i] = RandomNumberGenerationService.getRandomFloatNumberExceptZero(-10, 10);
            outputs[i] = intercept + (slope * inputs[i]);
            pairsOfInputsOutputsTrainingData.add(new Pair<>(inputs[i], outputs[i]));
        }
        LinearRegression linearRegression = new LinearRegression(pairsOfInputsOutputsTrainingData, modelError);
        linearRegression.train();
        assertTrue(Math.abs(intercept - linearRegression.getIntercept()) < 0.00001f);
        assertTrue(Math.abs(slope - linearRegression.getSlope()) < 0.00001f);
        for(int i = 0; i < 10; i++)
        {
            float x = RandomNumberGenerationService.getRandomFloatNumberExceptZero(-10, 10);
            assertTrue(Math.abs(modelFunction.run(x) - linearRegression.run(x)) < 0.00001f);
        }
    }
}
