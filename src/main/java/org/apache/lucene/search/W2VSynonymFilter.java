package org.apache.lucene.search;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.synonym.SynonymFilter;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.tokenattributes.TypeAttribute;
import org.apache.lucene.util.CharsRef;
import org.apache.lucene.util.CharsRefBuilder;
import org.deeplearning4j.models.word2vec.Word2Vec;

import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTagger;
import opennlp.tools.postag.POSTaggerME;

/**
 * A word2vec based synonym filter
 */
public final class W2VSynonymFilter extends TokenFilter {

  private final CharTermAttribute termAtt = addAttribute(CharTermAttribute.class);
  private final TypeAttribute typeAtt = addAttribute(TypeAttribute.class);
  private final PositionIncrementAttribute positionIncrementAttribute = addAttribute(PositionIncrementAttribute.class);

  private final Word2Vec word2Vec;
  private final List<PendingOutput> outputs = new LinkedList<>();
  private int positions = 0;
  private final POSTagger posTagger;

  W2VSynonymFilter(TokenStream input, Word2Vec word2Vec) {
    super(input);
    this.word2Vec = word2Vec;
    POSModel model;
    try {
      model = new POSModel(new FileInputStream(new File("/Users/teofili/dev/DC-FlinkMeetup/src/main/resources/opennlp-models/en-pos-maxent.bin")));
    } catch (IOException e) {
      throw new RuntimeException();
    }
    this.posTagger = new POSTaggerME(model);
  }

  @Override
  public boolean incrementToken() throws IOException {

    if (!outputs.isEmpty()) {
      PendingOutput output = outputs.remove(0);

      restoreState(output.state);

      termAtt.copyBuffer(output.charsRef.chars, output.charsRef.offset, output.charsRef.length);

      typeAtt.setType(SynonymFilter.TYPE_SYNONYM);
      positionIncrementAttribute.setPositionIncrement(output.posIncr);
      return true;
    }

    String type = typeAtt.type();
    if (!SynonymFilter.TYPE_SYNONYM.equals(type)) {
      positions++;
      positionIncrementAttribute.setPositionIncrement(positions);
      String word = termAtt.toString().trim();
      if (word.length() > 2) {
//        String[] tag = posTagger.tag(new String[] {word});
//
//        if (tag.length > 0 && ("NN".equals(tag[0]) || "JJ".equals(tag[0]) || "NNS".equals(tag[0]))) {
          Collection<String> list = word2Vec.wordsNearest(word, 2);
          for (String syn : list) {
            if (!syn.equals(word)) {
              CharsRefBuilder charsRefBuilder = new CharsRefBuilder();
              CharsRef cr = charsRefBuilder.append(syn).get();

              State state = captureState();
              outputs.add(new PendingOutput(state, cr, positions));
            }
          }
//        }
      }
    }
    return !outputs.isEmpty() || input.incrementToken();

  }

  @Override
  public void end() throws IOException {
    super.end();
    outputs.clear();
    positions = 0;
  }

  private class PendingOutput {
    private final State state;
    private final CharsRef charsRef;
    private final int posIncr;

    private PendingOutput(State state, CharsRef charsRef, int posIncr) {
      this.state = state;
      this.charsRef = charsRef;
      this.posIncr = posIncr;
    }
  }
}
