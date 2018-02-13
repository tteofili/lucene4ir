package org.apache.lucene.search;

import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;

/**
 *
 */
public class BinaryEncodingField extends Field {

  private static FieldType ft;

  static {
    ft = new FieldType();
    ft.setStored(true);
    ft.setTokenized(false);
  }

  private final BinaryEncoder binaryEncoder;

  public BinaryEncodingField(String name, BinaryEncoder binaryEncoder) {
    super(name, ft);
    this.binaryEncoder = binaryEncoder;
  }

  public BinaryEncodingField(String name, String value, BinaryEncoder encoder) throws Exception {
    super(name, encoder.encode(value), ft);
    binaryEncoder = encoder;
  }

  public interface BinaryEncoder {
    byte[] encode(String value) throws Exception;
  }

  @Override
  public void setStringValue(String value) {
    try {
      byte[] bytes = binaryEncoder.encode(value);
      setBytesValue(bytes);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

}
