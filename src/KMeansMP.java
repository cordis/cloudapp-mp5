import java.util.regex.Pattern;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;

import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;


public final class KMeansMP {
    private static Pattern DELIMITER = Pattern.compile(",");

    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: KMeansMP <input_file> <results>");
            System.exit(1);
        }
        String inputFile = args[0];
        String results_path = args[1];

        int k = 4;
        int iterations = 100;
        int runs = 1;
        long seed = 0;

        SparkConf sparkConf = new SparkConf().setAppName("KMeans MP");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        JavaRDD<String> lineList = sc.textFile(inputFile);

        JavaRDD<Vector> pointList = lineList.map(new ParsePoint());
        JavaRDD<String> labelList = lineList.map(new ParseLabel());
        KMeansModel model = KMeans.train(pointList.rdd(), k, iterations, runs, KMeans.RANDOM(), seed);
        labelList
                .zip(pointList)
                .mapToPair(new Categorize(model))
                .groupByKey()
                .saveAsTextFile(results_path);

        sc.stop();
    }

    private static class ParsePoint implements Function<String, Vector> {
        @Override
        public Vector call(String line) throws Exception {
            String[] tokenList = DELIMITER.split(line);
            double[] point = new double[tokenList.length - 1];
            for (int i = 1; i < tokenList.length; i++) {
                point[i - 1] = Double.parseDouble(tokenList[i]);
            }
            return Vectors.dense(point);
        }
    }

    private static class ParseLabel implements Function<String, String> {
        @Override
        public String call(String line) throws Exception {
            return DELIMITER.split(line, 1)[0];
        }
    }

    private static class Categorize implements PairFunction<Tuple2<String, Vector>, Integer, String> {
        private final KMeansModel model;

        public Categorize(KMeansModel model) {
            this.model = model;
        }

        @Override
        public Tuple2<Integer, String> call(Tuple2<String, Vector> stringVectorTuple2) throws Exception {
            String label = stringVectorTuple2._1();
            Vector point = stringVectorTuple2._2();
            Integer category = model.predict(point);
            return new Tuple2<>(category, label);
        }
    }
}
