package lucene4ir.similarity;

import java.io.IOException;
import java.util.Arrays;

import org.apache.lucene.index.FieldInvertState;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.MultiFields;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
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

  public static enum Smoothing {
    MEAN,
    IDF,
    TF,
    TF_IDF
  }

  private final Word2Vec word2Vec;
  private final String fieldName;
  private final Smoothing smoothing;

  public WordEmbeddingsSimilarity(Word2Vec word2Vec, String fieldName, Smoothing smoothing) {
    this.word2Vec = word2Vec;
    this.fieldName = fieldName;
    this.smoothing = smoothing;
  }

  public WordEmbeddingsSimilarity(Word2Vec word2Vec, String fieldName) {
    this.word2Vec = word2Vec;
    this.fieldName = fieldName;
    this.smoothing = Smoothing.TF;
  }

  @Override
  public long computeNorm(FieldInvertState state) {
    return 1l;
  }

  @Override
  public SimWeight computeWeight(CollectionStatistics collectionStats,
                                 TermStatistics... termStats) {
    return new EmbeddingsSimWeight(1f, collectionStats, termStats);
  }

  @Override
  public SimScorer simScorer(SimWeight weight, LeafReaderContext context) throws IOException {
    return new EmbeddingsSimScorer(weight, context);
  }

  private class EmbeddingsSimScorer extends SimScorer {
    private final EmbeddingsSimWeight weight;
    private final LeafReaderContext context;
    private Terms fieldTerms;
    private LeafReader reader;

    public EmbeddingsSimScorer(SimWeight weight, LeafReaderContext context) {
      this.weight = (EmbeddingsSimWeight) weight;
      this.context = context;
      this.reader = context.reader();
    }

    @Override
    public String toString() {
      return "EmbeddingsSimScorer{" +
          "weight=" + weight +
          ", context=" + context +
          ", fieldTerms=" + fieldTerms +
          ", reader=" + reader +
          '}';
    }

    @Override
    public float score(int doc, float freq) {
      try {
        INDArray denseQueryVector = getQueryVector();
        INDArray denseDocumentVector = VectorizeUtils.toDenseAverageVector(
            reader.getTermVector(doc, fieldName), reader.numDocs(), word2Vec, smoothing);
        return (float) Transforms.cosineSim(denseQueryVector, denseDocumentVector);
      } catch (IOException e) {
        return 0f;
      }
    }

    private INDArray getQueryVector() throws IOException {
      INDArray denseQueryVector = Nd4j.zeros(word2Vec.getLayerSize());
      String[] queryTerms = new String[weight.termStats.length];
      int i = 0;
      for (TermStatistics termStats : weight.termStats) {
        queryTerms[i] = termStats.term().utf8ToString();
        i++;
      }

      if (fieldTerms == null) {
        fieldTerms = MultiFields.getTerms(reader, fieldName);
      }

      for (String queryTerm : queryTerms) {
        TermsEnum iterator = fieldTerms.iterator();
        BytesRef term;
        while ((term = iterator.next()) != null) {
          TermsEnum.SeekStatus seekStatus = iterator.seekCeil(term);
          if (seekStatus.equals(TermsEnum.SeekStatus.END)) {
            iterator = fieldTerms.iterator();
          }
          if (seekStatus.equals(TermsEnum.SeekStatus.FOUND)) {
            String string = term.utf8ToString();
            if (string.equals(queryTerm)) {
              INDArray vector = word2Vec.getLookupTable().vector(queryTerm);
              if (vector != null) {
                double tf = iterator.totalTermFreq();
                double docFreq = iterator.docFreq();
                double smooth;
                switch (smoothing) {
                  case MEAN:
                    smooth = queryTerms.length;
                    break;
                  case TF:
                    smooth = tf;
                    break;
                  case IDF:
                    smooth = docFreq;
                    break;
                  case TF_IDF:
                    smooth = VectorizeUtils.tfIdf(reader.numDocs(), tf, docFreq);
                    break;
                  default:
                    smooth = VectorizeUtils.tfIdf(reader.numDocs(), tf, docFreq);
                }
                denseQueryVector.addi(vector.div(smooth));
              }
              break;
            }
          }
        }
      }
      return denseQueryVector;
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

    public EmbeddingsSimWeight(float boost, CollectionStatistics collectionStats, TermStatistics[] termStats) {
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

    @Override
    public float getValueForNormalization() {
      return 1f;
    }

    @Override
    public void normalize(float queryNorm, float boost) {

    }
  }
}
