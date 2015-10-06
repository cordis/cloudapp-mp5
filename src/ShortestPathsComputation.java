import org.apache.giraph.graph.BasicComputation;
import org.apache.giraph.conf.LongConfOption;
import org.apache.giraph.graph.Vertex;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;

import java.io.IOException;

/**
 * Compute shortest paths from a given source.
 */
public class ShortestPathsComputation extends BasicComputation<IntWritable, IntWritable, NullWritable, IntWritable> {
    public static final LongConfOption SOURCE_ID = new LongConfOption("SimpleShortestPathsVertex.sourceId", 1, "The shortest paths id");

    @Override
    public void compute(Vertex<IntWritable, IntWritable, NullWritable> vertex, Iterable<IntWritable> messages) throws IOException {
        if (this.getSuperstep() == 0) {
            vertex = this.setMinDistance(vertex, Integer.MAX_VALUE);
        }
        Integer minDistance = this.pullMinDistance(this.getMinDistance(vertex), messages);
        if (minDistance < vertex.getValue().get()) {
            vertex = this.setMinDistance(vertex, minDistance);
            vertex = this.pushMinDistance(vertex);
        }
        vertex.voteToHalt();
    }

    private Integer getMinDistance(Vertex<IntWritable, IntWritable, NullWritable> vertex) {
        if (this.isSource(vertex)) {
            return 0;
        }
        else {
            return Integer.MAX_VALUE;
        }
    }

    private boolean isSource(Vertex<IntWritable, ?, ?> vertex) {
        return vertex.getId().get() == SOURCE_ID.get(getConf());
    }

    private Integer pullMinDistance(Integer minDistance, Iterable<IntWritable> messageList) {
        for (IntWritable message: messageList) {
            minDistance = Math.min(minDistance, message.get());
        }
        return minDistance;
    }

    private Vertex<IntWritable, IntWritable, NullWritable> setMinDistance(Vertex<IntWritable, IntWritable, NullWritable> vertex, Integer minDistance) {
        vertex.setValue(new IntWritable(minDistance));
        return vertex;
    }

    private Vertex<IntWritable, IntWritable, NullWritable> pushMinDistance(Vertex<IntWritable, IntWritable, NullWritable> vertex) {
        sendMessageToAllEdges(vertex, new IntWritable(vertex.getValue().get() + 1));
        return vertex;
    }

}
