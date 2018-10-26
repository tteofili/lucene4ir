package lucene4ir.similarity;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

import lucene4ir.Lucene4IRConstants;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.core.LowerCaseFilterFactory;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.analysis.standard.StandardTokenizerFactory;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.*;
import org.apache.lucene.search.CollectionStatistics;
import org.apache.lucene.search.TermStatistics;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.util.BytesRef;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 *
 */
public class WordEmbeddingsSimilarity extends Similarity {

  public enum Smoothing {
    MEAN,
    IDF,
    TF,
    TF_IDF
  }

  private final Word2Vec word2Vec;
  private final String fieldName;
  private final Smoothing smoothing;
  private final Analyzer analyzer;

  public WordEmbeddingsSimilarity(Word2Vec word2Vec, String fieldName, Smoothing smoothing) {
    this.word2Vec = word2Vec;
    this.fieldName = fieldName;
    this.smoothing = smoothing;
    try {
      this.analyzer = CustomAnalyzer.builder().withTokenizer(StandardTokenizerFactory.class).addTokenFilter(LowerCaseFilterFactory.class).build();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public WordEmbeddingsSimilarity(Word2Vec word2Vec, String fieldName) {
    this(word2Vec, fieldName, Smoothing.TF_IDF);
  }

  @Override
  public long computeNorm(FieldInvertState state) {
    return 1l;
  }

  @Override
  public SimWeight computeWeight(float boost, CollectionStatistics collectionStats,
                                 TermStatistics... termStats) {
    return new EmbeddingsSimWeight(boost, collectionStats, termStats);
  }

  @Override
  public SimScorer simScorer(SimWeight weight, LeafReaderContext context) {
    return new EmbeddingsSimScorer(weight, context);
  }

  private class EmbeddingsSimScorer extends SimScorer {
    private INDArray queryVector = null;
    private final EmbeddingsSimWeight weight;
    private final LeafReaderContext context;
    private LeafReader reader;

    EmbeddingsSimScorer(SimWeight weight, LeafReaderContext context) {
      this.weight = (EmbeddingsSimWeight) weight;
      this.context = context;
      this.reader = context.reader();
    }

    @Override
    public String toString() {
      return "EmbeddingsSimScorer{" +
          "weight=" + weight +
          ", context=" + context +
          ", reader=" + reader +
          '}';
    }

    @Override
    public float score(int doc, float freq) {
      try {
        return (float) Transforms.cosineSim(getQueryVector(),
                VectorizeUtils.averageWordVectors(reader, doc, Lucene4IRConstants.FIELD_VECTOR, fieldName,
                        analyzer, word2Vec, smoothing));
      } catch (Exception e) {
        return 0f;
      }
    }

    private INDArray getQueryVector() throws IOException {
//      if (queryVector == null) {
//        List<String> queryTerms = new LinkedList<>();
//        for (TermStatistics termStats : weight.termStats) {
//          BytesRef term = termStats.term();
//          if (term != null) {
//            queryTerms.add(term.utf8ToString());
//          }
//        }
//        queryVector = VectorizeUtils.averageWordVectors(queryTerms, word2Vec.getLookupTable());
//      }
//      return queryVector;
      if (queryVector == null) {
        List<String> queryTerms = new LinkedList<>();
        for (TermStatistics termStats : weight.termStats) {
          BytesRef term = termStats.term();
          if (term != null) {
            queryTerms.add(term.utf8ToString());
          }
        }
        queryVector = VectorizeUtils.averageWordVectors(reader, fieldName, queryTerms, word2Vec, smoothing);
      }
      return queryVector;
    }



      @Override
    public float computeSlopFactor(int distance) {
      return 1;
    }

    @Override
    public float computePayloadFactor(int doc, int start, int end, BytesRef payload) {
      return 1;
    }
  }

  private class EmbeddingsSimWeight extends SimWeight {
    private final float boost;
    private final CollectionStatistics collectionStats;
    private final TermStatistics[] termStats;

    EmbeddingsSimWeight(float boost, CollectionStatistics collectionStats, TermStatistics[] termStats) {
      this.boost = boost;
      this.collectionStats = collectionStats;
      this.termStats = termStats;
    }

    @Override
    public String toString() {
      return "EmbeddingsSimWeight{" +
          "boost=" + boost +
          ", collectionStats=" + collectionStats +
          ", termStats=" + Arrays.toString(termStats) +
          '}';
    }
  }
}
