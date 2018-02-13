package lucene4ir;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileFilter;
import java.io.FileOutputStream;
import java.io.InputStreamReader;

import org.apache.commons.io.IOUtils;

/**
 *
 */
public class EmbeddingsQueryExpansionExperimentsApp {

  private static String[] measures = new String[] {"ndcg", "map", "gm_map", "Rprec", "bpref", "recip_rank"};
  private static String[] directories = new String[] {
      "/Users/teofili/Desktop/affect-dl/ootb-models",
      "/Users/teofili/Desktop/affect-dl/pre/",
      "/Users/teofili/Desktop/affect-dl/post/glove_post_training_bin",
      "/Users/teofili/Desktop/affect-dl/post/word2vec_post_training_bin",
      "/Users/teofili/Desktop/affect-dl/post/paragram_post_training_bin"
  };

  public static void main(String[] args) throws Exception {
    IndexerApp.main(new String[] {"params/index/index_params.xml"});

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
        RetrievalApp.main(new String[] {"params/retrieval_params.xml"});
        ExampleStatsApp.main(new String[] {"params/example_stats_params.xml"});

        builder.append(model.getName()).append(',');

        StringBuilder metrics = new StringBuilder();
        for (String measure : measures) {
          Command obj = new Command();

          //in mac oxs
          String command = "/Users/teofili/programs/trec_eval.9.0/trec_eval /Users/teofili/dev/lucene4ir/data/cacm/cacm.qrels /Users/teofili/dev/lucene4ir/data/cacm/bm25_results.res -m " + measure;

          String output = obj.execute(command);
          int beginIndex = output.indexOf("all") + 4;
          int endIndex = beginIndex + 6;
          output = output.substring(beginIndex, endIndex);

          metrics.append(output).append(',');
        }

        String line = metrics.toString();

        System.out.println(line);

        builder.append(line).append('\n');
      }
    }

    IOUtils.write(builder.toString(), new FileOutputStream(new File("/Users/teofili/Desktop/affect-dl", "output.csv")));

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