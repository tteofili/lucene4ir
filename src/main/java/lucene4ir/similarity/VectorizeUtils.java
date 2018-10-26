/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package lucene4ir.similarity;

import java.io.IOException;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.MultiFields;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.util.BytesRef;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static lucene4ir.similarity.WordEmbeddingsSimilarity.Smoothing.MEAN;

/**
 * utility class for converting Lucene {@link org.apache.lucene.document.Document}s to <code>Double</code> vectors.
 */
public class VectorizeUtils {

  private VectorizeUtils() {
    // no public constructors
  }

  /**
   * create a sparse <code>Double</code> vector given doc and field term vectors using TF-IDF
   *
   * @param docTerms   term vectors for a given document
   * @param fieldTerms field term vectors
   * @return a TF-IDF sparse vector
   * @throws IOException in case accessing the underlying index fails
   */
  public static double[] toSparseTFIDFDoubleArray(Terms docTerms, Terms fieldTerms, double n) throws IOException {
    TermsEnum fieldTermsEnum = fieldTerms.iterator();
    double[] tfIdfVector = null;
    if (docTerms != null && fieldTerms.size() > -1) {
      tfIdfVector = new double[(int) fieldTerms.size()];
      int i = 0;
      TermsEnum docTermsEnum = docTerms.iterator();
      BytesRef term;
      while ((term = fieldTermsEnum.next()) != null) {
        TermsEnum.SeekStatus seekStatus = docTermsEnum.seekCeil(term);
        if (seekStatus.equals(TermsEnum.SeekStatus.END)) {
          docTermsEnum = docTerms.iterator();
        }
        if (seekStatus.equals(TermsEnum.SeekStatus.FOUND)) {
          //weight(term) = (1+log(tf(term)))*log(N/df(term))
          long termFreq = docTermsEnum.totalTermFreq();
          int docFreq = fieldTermsEnum.docFreq();
          double tfIdf = tfIdf(n, termFreq, docFreq);
          tfIdfVector[i] = tfIdf;
        } else {
          tfIdfVector[i] = 0d;
        }
        i++;
      }
    }
    return tfIdfVector;
  }

  public static INDArray toDenseAverageTFIDFVector(Terms docTerms, double n, Word2Vec word2Vec) throws IOException {
    INDArray vector = Nd4j.zeros(word2Vec.getLayerSize());
    if (docTerms != null) {
      TermsEnum docTermsEnum = docTerms.iterator();
      BytesRef term;
      while ((term = docTermsEnum.next()) != null) {
        long termFreq = docTermsEnum.totalTermFreq();
        int docFreq = docTermsEnum.docFreq();
        double tfIdf = tfIdf(n, termFreq, docFreq);
        INDArray wordVector = word2Vec.getLookupTable().vector(term.utf8ToString()).div(tfIdf);
        vector.addi(wordVector);
      }
    }
    return vector;
  }

  public static INDArray averageWordVectors(Terms docTerms, double n, Word2Vec word2Vec, WordEmbeddingsSimilarity.Smoothing smoothing) throws IOException {
    INDArray vector = Nd4j.zeros(word2Vec.getLayerSize());
    if (docTerms != null) {
      TermsEnum docTermsEnum = docTerms.iterator();
      BytesRef term;
      while ((term = docTermsEnum.next()) != null) {
        INDArray wordVector = word2Vec.getLookupTable().vector(term.utf8ToString());
        if (wordVector != null) {
          long termFreq = docTermsEnum.totalTermFreq();
          int docFreq = docTermsEnum.docFreq();

          double smooth;
          switch (smoothing) {
            case MEAN:
              smooth = docTerms.size();
              break;
            case TF:
              smooth = termFreq;
              break;
            case IDF:
              smooth = docFreq;
              break;
            case TF_IDF:
              smooth = VectorizeUtils.tfIdf(n, termFreq, docFreq);
              break;
            default:
              smooth = VectorizeUtils.tfIdf(n, termFreq, docFreq);
          }
          vector.addi(wordVector.div(smooth));
        }
      }
    }
    return vector;
  }

  public static INDArray averageWordVectors(IndexReader reader, int doc, String vectorField, String contentField, Analyzer analyzer,
                                            Word2Vec word2Vec, WordEmbeddingsSimilarity.Smoothing smoothing) throws IOException {
    INDArray vector;
    Document document = reader.document(doc);
    if (document != null) {
      BytesRef binaryValue = document.getBinaryValue(vectorField);
      if (binaryValue != null) {
        vector = Nd4j.fromByteArray(binaryValue.bytes);
      } else {
        Terms termVector = reader.getTermVector(doc, contentField);
        if (termVector != null) {
          vector = averageWordVectors(termVector, reader.numDocs(), word2Vec, smoothing);
        }else {
          vector = averageWordVectors(getTokens(analyzer, contentField, document.get(contentField)), word2Vec);
        }
      }
    } else {
      vector = Nd4j.zeros(word2Vec.getLayerSize());
    }
    return vector;
  }

  public static INDArray averageWordVectors(Terms terms, Word2Vec word2Vec) throws IOException {
    Collection<String> indArrayCollection = new LinkedList<>();
    if (terms != null && terms.size() > -1) {
      TermsEnum docTermsEnum = terms.iterator();
      BytesRef term;
      while ((term = docTermsEnum.next()) != null) {
        TermsEnum.SeekStatus seekStatus = docTermsEnum.seekCeil(term);
        if (seekStatus.equals(TermsEnum.SeekStatus.END)) {
          docTermsEnum = terms.iterator();
        }
        if (seekStatus.equals(TermsEnum.SeekStatus.FOUND)) {
          indArrayCollection.add(term.utf8ToString());
        }

      }
    }
    return word2Vec.getWordVectorsMean(indArrayCollection);
  }

  public static double tfIdf(double n, double termFreq, double docFreq) {
    return 1 + Math.log(termFreq) * Math.log(n / docFreq);
  }

  public static double cosineSimilarity(double[] vectorA, double[] vectorB) {
    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;
    for (int i = 0; i < vectorA.length; i++) {
      dotProduct += vectorA[i] * vectorB[i];
      normA += Math.pow(vectorA[i], 2);
      normB += Math.pow(vectorB[i], 2);
    }
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  public static INDArray averageWordVectors(Collection<String> words, Word2Vec word2Vec) {
    INDArray denseDocumentVector;
    try {
      denseDocumentVector = word2Vec.getWordVectorsMean(words);
    } catch (Exception e) {
      denseDocumentVector = Nd4j.zeros(word2Vec.getLayerSize());
      int i = 0;
      for (String token : words) {
        INDArray wordVector = word2Vec.getLookupTable().vector(token);
        if (wordVector != null) {
          denseDocumentVector.addi(wordVector);
          i++;
        }
        INDArray unkVector = word2Vec.getLookupTable().vector(word2Vec.getUNK());
        if (unkVector != null) {
          denseDocumentVector.addi(unkVector);
          i++;
        }
      }
      denseDocumentVector.divi(i);
    }
    return denseDocumentVector;
  }

  public static INDArray averageWordVectors(Collection<String> words, WeightLookupTable lookupTable) {
    if (words.isEmpty()) {
      return Nd4j.zeros(lookupTable.layerSize());
    }
    INDArray denseDocumentVector = Nd4j.zeros(words.size(), lookupTable.layerSize());
    int i = 0;
    for (String w : words) {
      INDArray vector = lookupTable.vector(w);
      if (vector == null) {
        vector = lookupTable.vector("UNK");
      }
      if (vector != null) {
        denseDocumentVector.putRow(i, vector);
      }
      i++;
    }
    return denseDocumentVector.add(1e-10).mean(0);
  }

  public static INDArray averageWV(String[] words, Word2Vec word2Vec) {
    INDArray denseQueryVector = Nd4j.zeros(word2Vec.getLayerSize());
    for (String term : words) {
      INDArray vector = word2Vec.getLookupTable().vector(term);
      if (vector != null) {
        denseQueryVector.addi(vector).divi(words.length);
      }
    }
    return denseQueryVector;
  }

  private static Collection<String> getTokens(Analyzer analyzer, String field, String text) throws IOException {
    Collection<String> tokens = new LinkedList<>();
    TokenStream ts = analyzer.tokenStream(field, text);
    ts.reset();
    ts.addAttribute(CharTermAttribute.class);
    while (ts.incrementToken()) {
      CharTermAttribute charTermAttribute = ts.getAttribute(CharTermAttribute.class);
      String token = new String(charTermAttribute.buffer(), 0, charTermAttribute.length());
      tokens.add(token);
    }
    ts.end();
    ts.close();
    return tokens;
  }

  public static INDArray averageWordVectors(IndexReader reader, String fieldName, Collection<String> queryTerms, Word2Vec word2Vec,
                                  WordEmbeddingsSimilarity.Smoothing smoothing) throws IOException {
    INDArray denseQueryVector = Nd4j.zeros(word2Vec.getLayerSize());

    Terms fieldTerms = MultiFields.getTerms(reader, fieldName);

    for (String queryTerm : queryTerms) {
      TermsEnum iterator = fieldTerms.iterator();
      boolean seekStatus = iterator.seekExact(new BytesRef(queryTerm));
      if (seekStatus) {
        INDArray vector = word2Vec.getLookupTable().vector(queryTerm);
        if (vector != null) {
          double tf = iterator.totalTermFreq();
          double docFreq = iterator.docFreq();
          double smooth;
          switch (smoothing) {
            case MEAN:
              smooth = queryTerms.size();
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
              smooth = queryTerms.size();
          }
          denseQueryVector.addi(vector).divi(smooth);
        }
      }
    }
    return denseQueryVector;
  }
}
