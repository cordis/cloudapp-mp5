import org.apache.giraph.graph.BasicComputation;
import org.apache.giraph.edge.Edge;
import org.apache.giraph.graph.Vertex;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;

import java.io.IOException;

/**
 * Implementation of the connected component algorithm that identifies
 * connected components and assigns each vertex its "component
 * identifier" (the smallest vertex id in the component).
 */
public class ConnectedComponentsComputation extends BasicComputation<IntWritable, IntWritable, NullWritable, IntWritable> {
    /**
     * Propagates the smallest vertex id to all neighbors. Will always choose to
     * halt and only reactivate if a smaller id has been sent to it.
     *
     * @param vertex   Vertex
     * @param messages Iterator of messages from the previous superstep.
     * @throws IOException
     */
    @Override
    public void compute(Vertex<IntWritable, IntWritable, NullWritable> vertex, Iterable<IntWritable> messages) throws IOException {
        if (getSuperstep() == 0) {
            Integer candidateMinVertexId = this.getInitialMinVertexId(vertex);
            if (candidateMinVertexId < vertex.getValue().get()) {
                vertex = this.setMinVertexId(vertex, candidateMinVertexId);
            }
            vertex = this.pushVertexId(vertex);
        }
        else {
            Integer candidateMinVertexId = this.pullMinVertexId(vertex.getValue().get(), messages);
            if (candidateMinVertexId < vertex.getValue().get()) {
                vertex = this.setMinVertexId(vertex, candidateMinVertexId);
                vertex = this.pushVertexId(vertex);
            }
        }
        vertex.voteToHalt();
    }

    private Integer getInitialMinVertexId(Vertex<IntWritable, IntWritable, NullWritable> vertex) {
        Integer minVertexId = vertex.getValue().get();
        for (Edge<IntWritable, NullWritable> edge : vertex.getEdges()) {
            minVertexId = Math.min(minVertexId, edge.getTargetVertexId().get());
        }
        return minVertexId;
    }

    private Integer pullMinVertexId(Integer minVertexId, Iterable<IntWritable> messageList) {
        for (IntWritable candidateMinVertexId: messageList) {
            minVertexId = Math.min(minVertexId, candidateMinVertexId.get());
        }
        return minVertexId;
    }

    private Vertex<IntWritable, IntWritable, NullWritable> setMinVertexId(Vertex<IntWritable, IntWritable, NullWritable> vertex, Integer candidateMinVertexId) {
        vertex.setValue(new IntWritable(candidateMinVertexId));
        return vertex;
    }

    private Vertex<IntWritable, IntWritable, NullWritable> pushVertexId(Vertex<IntWritable, IntWritable, NullWritable> vertex) {
        sendMessageToAllEdges(vertex, vertex.getValue());
        return vertex;
    }

}
