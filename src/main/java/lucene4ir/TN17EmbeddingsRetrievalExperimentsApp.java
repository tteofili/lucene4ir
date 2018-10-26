package lucene4ir;

import org.apache.commons.io.IOUtils;
import org.apache.lucene.search.similarities.*;

import java.io.*;

/**
 *
 */
public class TN17EmbeddingsRetrievalExperimentsApp {

  private static String[] measures = new String[] {"ndcg", "map", "Rprec", "recip_rank", "P"};

  private static Similarity[] similarities = new Similarity[]{new ClassicSimilarity(), new BM25Similarity(),
          new LMDirichletSimilarity(), new LMJelinekMercerSimilarity(0.1f)};

  private static String[] directories = new String[] {
          "/Users/teofili/Desktop/tests/affect-dl/pre/",
          "/Users/teofili/Desktop/tests/affect-dl/ootb-models",
          "/Users/teofili/Desktop/tests/affect-dl/post/glove_post_training_bin",
          "/Users/teofili/Desktop/tests/affect-dl/post/word2vec_post_training_bin",
          "/Users/teofili/Desktop/tests/affect-dl/post/paragram_post_training_bin"
  };

  public static void main(String[] args) throws Exception {
//    IndexerApp.main(new String[] {"params/index/index_params_tn.xml"});

    StringBuilder builder = new StringBuilder();
    builder.append("model,");

    for (String measure : measures) {
      builder.append(measure).append(',');
    }
    builder.append("\n");

    for (Similarity s : similarities) {
      RetrievalApp ret = new RetrievalApp("params/retrieval_params_tn.xml");
      ret.simfn = s;
      ret.searcher.setSimilarity(ret.simfn);
//      System.out.println("running classic retrieval "+ret);
      runRetrieval(s.toString(), builder, ret);
    }

    for (String directory : directories) {
      FileFilter filter = pathname -> !pathname.getName().startsWith(".");
      File modelsDirectory = new File(directory);
      File[] models = modelsDirectory.listFiles(filter);

      builder.append("\n");

      for (File model : models) {
        String absolutePath = model.getAbsolutePath();
        System.out.println("using model " + absolutePath);
        System.setProperty("model", absolutePath);
        RetrievalApp ret = new RerankingRetrievalApp("params/retrieval_params_tn.xml");
//        System.out.println("running mixed retrieval " + ret);
//        RetrievalApp ret = new RetrievalApp("params/retrieval_params_tn.xml");
//        runRetrieval(model.getName(), builder, ret);

        for (Similarity s : similarities) {
          ret.simfn = s;
          ret.searcher.setSimilarity(ret.simfn);
          runRetrieval(model.getName()+" "+s, builder, ret);
        }


      }
    }

    IOUtils.write(builder.toString(), new FileOutputStream(new File("/Users/teofili/Desktop/tests/affect-dl", "output.csv")));

  }

  private static void runRetrieval(String tag, StringBuilder builder, RetrievalApp ret) throws IOException {
    ret.processQueryFile();
//    ExampleStatsApp.main(new String[] {"params/example_stats_params_tn.xml"});

    builder.append(tag).append(',');

    StringBuilder metrics = new StringBuilder();
    for (String measure : measures) {
      Command obj = new Command();

      //in mac oxs
      String command = "/Users/teofili/programs/trec_eval.9.0/trec_eval /Users/teofili/dev/lucene4ir/data/tn17/qrels.txt /Users/teofili/dev/lucene4ir/data/tn17/wv_results.res -m " + measure;

      String output = obj.execute(command);
      try {
        int beginIndex = output.indexOf("all") + 4;
        int endIndex = beginIndex + 6;
        output = output.substring(beginIndex, endIndex);
      } catch (Throwable t) {
        // do nothing
      }

      metrics.append(output).append(',');
    }

    String line = metrics.toString();

    builder.append(line).append('\n');
    System.out.println(builder);
  }

  private static class Command {

    private String execute(String command) {

      StringBuilder output = new StringBuilder();

      Process p;
      try {
        p = Runtime.getRuntime().exec(command);
        p.waitFor();
        BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));

        String line;
        while ((line = reader.readLine()) != null) {
          output.append(line).append('\n');
        }

      } catch (Exception e) {
        e.printStackTrace();
      }

      return output.toString();

    }
  }
}