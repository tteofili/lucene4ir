package lucene4ir;

import lucene4ir.similarity.*;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.index.MultiFields;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.*;
import java.util.*;

public class RerankingRetrievalApp extends RetrievalApp {

    private Word2Vec vec;

    public RerankingRetrievalApp(String retrievalParamFile) {
        super(retrievalParamFile);
        if (System.getProperty("model") != null) {
            String model = System.getProperty("model");
            try {
                vec = WordVectorSerializer.readWord2VecModel(model);
            } catch (Exception e) {
                try {
                    vec = WordVectorSerializer.readBinaryModel(new File(model), false, false);
                } catch (Throwable e1) {
                    System.err.println("failed loading model "+model);
                    e1.printStackTrace();
                }
            }
        }
    }

    public ScoreDoc[] runQuery(String qno, String queryTerms){
        ScoreDoc[] hits = null;

        parser.setAnalyzer(analyzer);
        System.out.println("Query No.: " + qno + " " + queryTerms);
        try {
            Query query = parser.parse(QueryParser.escape(queryTerms), null);

            try {
                TopDocs results = searcher.search(query, p.maxResults * 10);
                if (results.scoreDocs.length > 0) {
                    kNNRerank(p.maxResults, results, queryTerms, Lucene4IRConstants.FIELD_CONTENT, false);
                }
                hits = results.scoreDocs;
            }
            catch (IOException ioe){
                ioe.printStackTrace();
                System.exit(1);
            }
        } catch (Exception pe){
            pe.printStackTrace();
            System.exit(1);
        }
        return hits;
    }

    private void kNNRerank(int k, TopDocs docs, String queryText, String fieldName, boolean exact) throws IOException {
        Collection<String> queryTerms = getTokens(analyzer, fieldName, queryText);
        WordEmbeddingsSimilarity.Smoothing smoothing = WordEmbeddingsSimilarity.Smoothing.TF_IDF;
        INDArray queryVector = VectorizeUtils.averageWordVectors(reader, fieldName, queryTerms, vec, smoothing);
        List<Integer> toDiscard = new LinkedList<>();
        for (int j = 0; j < docs.scoreDocs.length; j++) {
            INDArray documentVector = VectorizeUtils.averageWordVectors(reader, docs.scoreDocs[j].doc, Lucene4IRConstants.FIELD_VECTOR,
                    fieldName, analyzer, vec, smoothing);

//            double similarity = 1 / (1e-10 + Transforms.euclideanDistance(queryVector, documentVector));
            double similarity = Transforms.cosineSim(queryVector, documentVector);
            if (Double.isNaN(similarity) || similarity < 0.5) {
                toDiscard.add(docs.scoreDocs[j].doc);
            }
            if (exact) {
                docs.scoreDocs[j].score = (float) similarity;
            } else {
                docs.scoreDocs[j].score += (float) similarity;
//                float score = (float) (docs.scoreDocs[j].score + similarity);
//                docs.scoreDocs[j].score = (float) ((docs.scoreDocs[j].score*0.5f + 0.5f*similarity*score)/score);
            }
        }
        if (!toDiscard.isEmpty()) {
            docs.scoreDocs = Arrays.stream(docs.scoreDocs).filter(e -> !toDiscard.contains(e.doc)).toArray(ScoreDoc[]::new);
        }

        Arrays.parallelSort(docs.scoreDocs, 0, docs.scoreDocs.length, (o1, o2) -> { // rerank scoreDocs
            return -1 * Double.compare(o1.score, o2.score);
        });
        if (docs.scoreDocs.length > k) {
            docs.scoreDocs = Arrays.copyOfRange(docs.scoreDocs, 0, k); // retain only the top k nearest neighbours
        }
        if (docs.scoreDocs.length > 0) {
            docs.setMaxScore(docs.scoreDocs[0].score);
        }
        docs.totalHits = docs.scoreDocs.length;
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

}