package org.apache.lucene.search;

import java.util.Map;

import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.util.TokenFilterFactory;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;

/**
 *
 */
public class W2VSynonymFilterFactory extends TokenFilterFactory {

  private final Word2Vec vec;

  public W2VSynonymFilterFactory(Map<String, String> args) {
    super(args);

    String model = args.get("model");
    if (System.getProperty("model") != null) {
      model = System.getProperty("model");
    }
    vec = WordVectorSerializer.readWord2VecModel(model);
  }

  @Override
  public TokenStream create(TokenStream tokenStream) {
    return new W2VSynonymFilter(tokenStream, vec, 0.9d);
  }

}
