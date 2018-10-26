package lucene4ir.indexer;

import com.google.common.base.Joiner;
import lucene4ir.Lucene4IRConstants;
import lucene4ir.similarity.VectorizeUtils;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.search.BinaryEncodingField;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.Collection;
import java.util.LinkedList;

public class NYTIndexer extends DocumentIndexer {

    private Field docnumField;
    private Field titleField;
    private Field textField;
    private Field authorField;
    private Field allField;
    private Field vectorField;
    private Field entitiesField;
    private Document doc;
    private WeightLookupTable lookupTable;
    private final NYTCorpusDocumentParser parser = new NYTCorpusDocumentParser();

    public NYTIndexer(String indexPath, String tokenFilterFile, boolean positional, WeightLookupTable lookupTable) throws Exception {
        super(indexPath, tokenFilterFile, positional);

        this.lookupTable = lookupTable;
        doc = new Document();

        initFields();
        initNEWSDoc();
    }

    private void initFields() throws Exception {
        docnumField = new StringField(Lucene4IRConstants.FIELD_DOCNUM, "", Field.Store.YES);
        if (indexPositions) {
            titleField = new TermVectorEnabledTextField(Lucene4IRConstants.FIELD_TITLE, "", Field.Store.YES);
            textField = new TermVectorEnabledTextField(Lucene4IRConstants.FIELD_CONTENT, "", Field.Store.YES);
            allField = new TermVectorEnabledTextField(Lucene4IRConstants.FIELD_ALL, "", Field.Store.NO);
            authorField = new TermVectorEnabledTextField(Lucene4IRConstants.FIELD_AUTHOR, "", Field.Store.NO);
            entitiesField = new TermVectorEnabledTextField(Lucene4IRConstants.FIELD_ENTITIES, "", Field.Store.YES);
        } else {
            titleField = new TextField(Lucene4IRConstants.FIELD_TITLE, "", Field.Store.YES);
            textField = new TextField(Lucene4IRConstants.FIELD_CONTENT, "", Field.Store.YES);
            allField = new TextField(Lucene4IRConstants.FIELD_ALL, "", Field.Store.NO);
            authorField = new TextField(Lucene4IRConstants.FIELD_AUTHOR, "", Field.Store.NO);
            entitiesField = new TextField(Lucene4IRConstants.FIELD_ENTITIES, "", Field.Store.YES);
        }
        if (lookupTable != null) {
            // average word vector binary encoder
            BinaryEncodingField.BinaryEncoder encoder = value -> {

                if (value != null && value.trim().length() > 0) {
                    Analyzer analyzer = new StandardAnalyzer();
                    TokenStream tokenStream = analyzer.tokenStream(null, value);
                    tokenStream.addAttribute(CharTermAttribute.class);
                    tokenStream.reset();
                    Collection<String> words = new LinkedList<>();
                    while (tokenStream.incrementToken()) {
                        CharTermAttribute attribute = tokenStream.getAttribute(CharTermAttribute.class);
                        String token = attribute.toString();
                        words.add(token);
                    }
                    INDArray vector = VectorizeUtils.averageWordVectors(words, lookupTable);

                    return Nd4j.toByteArray(vector);
                } else {
                    return Nd4j.toByteArray(Nd4j.zeros(1, lookupTable.layerSize()));
                }
            };
            vectorField = new BinaryEncodingField(Lucene4IRConstants.FIELD_VECTOR, "", encoder);
        }
    }

    private void initNEWSDoc() {
        doc.add(docnumField);
        doc.add(titleField);
        doc.add(textField);
        doc.add(allField);
        doc.add(authorField);
        doc.add(entitiesField);
        if (lookupTable != null) {
            doc.add(vectorField);
        }
    }

    public Document createNEWSDocument(String docid, String author, String title, String content, String entities, String all) {
        doc.clear();

        docnumField.setStringValue(docid);
        titleField.setStringValue(title);
        allField.setStringValue(all);
        textField.setStringValue(content);
        authorField.setStringValue(author);
        entitiesField.setStringValue(entities);
        if (lookupTable != null) {
            vectorField.setStringValue(content);
        }

        doc.add(docnumField);
        doc.add(authorField);
        doc.add(titleField);
        doc.add(textField);
        doc.add(entitiesField);
        doc.add(allField);
        if (lookupTable != null) {
            doc.add(vectorField);
        }
        return doc;
    }

    public void indexDocumentsFromFile(String filename) {
        File f = new File(filename);
        NYTCorpusDocument nytCorpusDocument = parser.parseNYTCorpusDocumentFromFile(f, false);
        String docid = String.valueOf(nytCorpusDocument.guid);
        String author = nytCorpusDocument.normalizedByline == null ? "" : nytCorpusDocument.normalizedByline;

        StringBuilder title = new StringBuilder();
        if (nytCorpusDocument.titles != null ) {
            for (String t : nytCorpusDocument.titles) {
                if (title.length() > 0) {
                    title.append(' ');
                }
                title.append(t);
            }
        }
        if (nytCorpusDocument.headline != null ) {
            title.append(' ').append(nytCorpusDocument.headline);
        }
        if (nytCorpusDocument.kicker != null ) {
            title.append(' ').append(nytCorpusDocument.kicker);
        }

        StringBuilder body = new StringBuilder();
        if (nytCorpusDocument.articleAbstract != null) {
            body.append(nytCorpusDocument.articleAbstract);
        }
        if (nytCorpusDocument.body != null) {
            if (body.length() > 0) {
                body.append(' ');
            }
            body.append(nytCorpusDocument.body);
        }

        StringBuilder other = new StringBuilder();
        for (String d : nytCorpusDocument.descriptors) {
            if (other.length() > 0) {
                other.append(' ');
            }
            other.append(d);
        }
        for (String d : nytCorpusDocument.biographicalCategories) {
            if (other.length() > 0) {
                other.append(' ');
            }
            other.append(d);
        }
        for (String d : nytCorpusDocument.taxonomicClassifiers) {
            if (other.length() > 0) {
                other.append(' ');
            }
            other.append(d);
        }
        for (String e : nytCorpusDocument.generalOnlineDescriptors) {
            if (other.length() > 0) {
                other.append(' ');
            }
            other.append(e);
        }
        for (String e : nytCorpusDocument.onlineDescriptors) {
            if (other.length() > 0) {
                other.append(' ');
            }
            other.append(e);
        }
        other.append(' ').append(nytCorpusDocument.section);

        for (String e : nytCorpusDocument.onlinePeople) {
            if (other.length() > 0) {
                other.append(' ');
            }
            other.append(e);
        }
        for (String e : nytCorpusDocument.onlineLocations) {
            if (other.length() > 0) {
                other.append(' ');
            }
            other.append(e);
        }
        for (String e : nytCorpusDocument.onlineOrganizations) {
            if (other.length() > 0) {
                other.append(' ');
            }
            other.append(e);
        }
        if (nytCorpusDocument.featurePage != null) {
            if (other.length() > 0) {
                other.append(nytCorpusDocument.featurePage);
            }
        }

        String all = Joiner.on(" ").join(author, title, body, other);
        createNEWSDocument(docid, author, title.toString(), body.toString(), other.toString(), all);
        addDocumentToIndex(doc);
    }
}
