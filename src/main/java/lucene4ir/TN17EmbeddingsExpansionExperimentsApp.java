package lucene4ir;

import org.apache.commons.io.IOUtils;

import java.io.*;

/**
 *
 */
public class TN17EmbeddingsExpansionExperimentsApp {

  private static String[] measures = new String[] {"ndcg", "map", "Rprec", "recip_rank", "P"};

  private static String[] directories = new String[] {
          "/Users/teofili/Desktop/tests/affect-dl/pre/",
          "/Users/teofili/Desktop/tests/affect-dl/ootb-models",
          "/Users/teofili/Desktop/tests/affect-dl/post/glove_post_training_bin",
          "/Users/teofili/Desktop/tests/affect-dl/post/word2vec_post_training_bin",
          "/Users/teofili/Desktop/tests/affect-dl/post/paragram_post_training_bin"
  };

  public static void main(String[] args) throws Exception {

    StringBuilder builder = new StringBuilder();
    builder.append("model,");

    for (String measure : measures) {
      builder.append(measure).append(',');
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
//        IndexerApp.main(new String[] {"params/index/index_params_tn.xml"});
        RetrievalApp ret = new RetrievalApp("params/retrieval_params_tn.xml");
        ret.processQueryFile();
//        ExampleStatsApp.main(new String[] {"params/example_stats_params_tn.xml"});

        builder.append(model.getName()).append(',');

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
    }

    IOUtils.write(builder.toString(), new FileOutputStream(new File("/Users/teofili/Desktop/tests/affect-dl", "output.csv")));

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