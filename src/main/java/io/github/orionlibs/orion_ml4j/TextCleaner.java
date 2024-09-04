package io.github.orionlibs.orion_ml4j;

public class TextCleaner
{
    public String clean(String text)
    {
        text = text.replaceAll("[^\\p{ASCII}]", "");
        text = text.replaceAll("\\s+", " ");
        text = text.replaceAll("\\p{Cntrl}", "");
        text = text.replaceAll("[^\\p{Print}]", "");
        text = text.replaceAll("\\p{C}", "");
        return text;
    }
}
