package com.booklore.tools.pdfwriter;

import org.apache.commons.lang3.StringUtils;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDDocumentInformation;
import org.apache.pdfbox.pdmodel.common.PDMetadata;
import org.apache.xmpbox.XMPMetadata;
import org.apache.xmpbox.schema.DublinCoreSchema;
import org.apache.xmpbox.xml.XmpSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.util.Calendar;
import java.util.GregorianCalendar;
import java.util.Objects;

public class PdfMetadataAugmenter {

    private static final Logger log = LoggerFactory.getLogger(PdfMetadataAugmenter.class);

    public void writeMetadata(File sourcePdf, File destinationPdf, BookMetadataInput metadata) throws IOException {
        if (metadata == null) {
            throw new IllegalArgumentException("Metadata JSON is empty");
        }
        if (!sourcePdf.exists()) {
            throw new IOException("Source PDF not found: " + sourcePdf.getAbsolutePath());
        }

        try (PDDocument pdf = Loader.loadPDF(sourcePdf)) {
            pdf.setAllSecurityToBeRemoved(true);
            applyMetadataToDocument(pdf, metadata);
            Path parent = destinationPdf.toPath().toAbsolutePath().getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            pdf.save(destinationPdf);
            log.info("Augmented PDF written to {}", destinationPdf.getAbsolutePath());
        } catch (IOException e) {
            throw e;
        } catch (Exception e) {
            throw new IOException("Failed to embed metadata: " + e.getMessage(), e);
        }
    }

    private void applyMetadataToDocument(PDDocument pdf, BookMetadataInput metadata) throws Exception {
        PDDocumentInformation info = pdf.getDocumentInformation();
        if (metadata.getTitle() != null) {
            info.setTitle(metadata.getTitle());
        }
        if (metadata.getPublisher() != null) {
            info.setProducer(metadata.getPublisher());
        }
        if (metadata.getAuthors() != null && !metadata.getAuthors().isEmpty()) {
            info.setAuthor(String.join(", ", metadata.getAuthors()));
        }
        if (metadata.getCategories() != null && !metadata.getCategories().isEmpty()) {
            info.setKeywords(String.join(", ", metadata.getCategories()));
        }

        XMPMetadata xmp = XMPMetadata.createXMPMetadata();
        DublinCoreSchema dc = xmp.createAndAddDublinCoreSchema();

        if (metadata.getTitle() != null) {
            dc.setTitle(metadata.getTitle());
        }
        if (metadata.getDescription() != null) {
            dc.setDescription(metadata.getDescription());
        }
        if (metadata.getPublisher() != null) {
            dc.addPublisher(metadata.getPublisher());
        }
        if (metadata.getLanguage() != null) {
            dc.addLanguage(metadata.getLanguage());
        }
        if (metadata.getPublishedDate() != null) {
            Calendar cal = GregorianCalendar.from(metadata.getPublishedDate().atStartOfDay(ZoneId.systemDefault()));
            dc.addDate(cal);
        }
        if (metadata.getAuthors() != null) {
            metadata.getAuthors().forEach(dc::addCreator);
        }
        if (metadata.getCategories() != null) {
            metadata.getCategories().forEach(dc::addSubject);
        }

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        new XmpSerializer().serialize(xmp, baos, true);
        byte[] baseXmpBytes = baos.toByteArray();

        byte[] newXmpBytes = addCustomIdentifiersToXmp(baseXmpBytes, metadata);

        byte[] existingXmpBytes = null;
        PDMetadata existing = pdf.getDocumentCatalog().getMetadata();
        if (existing != null) {
            try {
                existingXmpBytes = existing.toByteArray();
            } catch (IOException ignore) {
            }
        }

        if (!isXmpMetadataDifferent(existingXmpBytes, newXmpBytes)) {
            log.info("XMP metadata unchanged, skipping write");
            return;
        }

        PDMetadata pdMetadata = new PDMetadata(pdf);
        pdMetadata.importXMPMetadata(newXmpBytes);
        pdf.getDocumentCatalog().setMetadata(pdMetadata);
    }

    private byte[] addCustomIdentifiersToXmp(byte[] xmpBytes, BookMetadataInput metadata) throws Exception {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        factory.setNamespaceAware(true);
        DocumentBuilder builder = factory.newDocumentBuilder();
        Document doc = builder.parse(new ByteArrayInputStream(xmpBytes));

        Element rdfRoot = (Element) doc.getElementsByTagNameNS("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "RDF")
                .item(0);
        if (rdfRoot == null) {
            throw new IllegalStateException("RDF root missing in XMP");
        }

        Element rdfDescription = doc.createElementNS("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf:Description");
        rdfDescription.setAttributeNS("http://www.w3.org/2000/xmlns/", "xmlns:xmp", "http://ns.adobe.com/xap/1.0/");
        rdfDescription.setAttributeNS("http://www.w3.org/2000/xmlns/", "xmlns:xmpidq",
                "http://ns.adobe.com/xmp/Identifier/qual/1.0/");
        rdfDescription.setAttributeNS("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf:about", "");

        Element xmpIdentifier = doc.createElementNS("http://ns.adobe.com/xap/1.0/", "xmp:Identifier");
        Element rdfBag = doc.createElementNS("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf:Bag");

        appendIdentifier(doc, rdfBag, "google", metadata.getGoogleId());
        appendIdentifier(doc, rdfBag, "goodreads", metadata.getGoodreadsId());
        appendIdentifier(doc, rdfBag, "comicvine", metadata.getComicvineId());
        appendIdentifier(doc, rdfBag, "hardcover", metadata.getHardcoverId());
        appendIdentifier(doc, rdfBag, "amazon", metadata.getAsin());
        appendIdentifier(doc, rdfBag, "isbn", StringUtils.defaultIfBlank(metadata.getIsbn13(), metadata.getIsbn10()));

        if (rdfBag.hasChildNodes()) {
            xmpIdentifier.appendChild(rdfBag);
            rdfDescription.appendChild(xmpIdentifier);
        }

        ZonedDateTime now = ZonedDateTime.now();
        rdfDescription.appendChild(createSimpleElement(doc, "xmp:MetadataDate", now.toString()));
        rdfDescription.appendChild(createSimpleElement(doc, "xmp:CreateDate",
                metadata.getPublishedDate() != null
                        ? metadata.getPublishedDate().atStartOfDay(ZoneId.systemDefault()).toString()
                        : now.toString()));
        rdfDescription.appendChild(createSimpleElement(doc, "xmp:CreatorTool", "Booklore"));
        rdfDescription.appendChild(createSimpleElement(doc, "xmp:ModifyDate", now.toString()));

        rdfRoot.appendChild(rdfDescription);

        if (metadata.getSeriesName() != null) {
            Element calibreDescription = doc.createElementNS("http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                    "rdf:Description");
            calibreDescription.setAttributeNS("http://www.w3.org/2000/xmlns/", "xmlns:calibre",
                    "http://calibre-ebook.com/xmp-namespace");
            calibreDescription.setAttributeNS("http://www.w3.org/2000/xmlns/", "xmlns:calibreSI",
                    "http://calibre-ebook.com/xmp-namespace-series-index");
            calibreDescription.setAttributeNS("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf:about", "");

            Element seriesElem = doc.createElementNS("http://calibre-ebook.com/xmp-namespace", "calibre:series");
            seriesElem.setAttributeNS("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf:parseType", "Resource");

            Element valueElem = doc.createElementNS("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf:value");
            valueElem.setTextContent(metadata.getSeriesName());
            seriesElem.appendChild(valueElem);

            Float seriesIndex = metadata.getSeriesNumber();
            Element indexElem = doc.createElementNS("http://calibre-ebook.com/xmp-namespace-series-index",
                    "calibreSI:series_index");
            indexElem.setTextContent(seriesIndex != null ? String.format("%.2f", seriesIndex) : "0.00");
            seriesElem.appendChild(indexElem);

            calibreDescription.appendChild(seriesElem);
            rdfRoot.appendChild(calibreDescription);
        }

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        Transformer tf = TransformerFactory.newInstance().newTransformer();
        tf.setOutputProperty(OutputKeys.OMIT_XML_DECLARATION, "yes");
        tf.setOutputProperty(OutputKeys.INDENT, "yes");
        tf.transform(new DOMSource(doc), new StreamResult(baos));
        return baos.toByteArray();
    }

    private void appendIdentifier(Document doc, Element bag, String scheme, String value) {
        if (StringUtils.isBlank(value)) {
            return;
        }
        Element li = doc.createElementNS("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf:li");
        li.setAttributeNS("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf:parseType", "Resource");

        Element schemeElem = doc.createElementNS("http://ns.adobe.com/xmp/Identifier/qual/1.0/", "xmpidq:Scheme");
        schemeElem.setTextContent(scheme);

        Element valueElem = doc.createElementNS("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf:value");
        valueElem.setTextContent(value);

        li.appendChild(schemeElem);
        li.appendChild(valueElem);
        bag.appendChild(li);
    }

    private Element createSimpleElement(Document doc, String name, String content) {
        String namespace = name.startsWith("calibre:")
                ? "http://calibre-ebook.com/xmp-namespace"
                : "http://ns.adobe.com/xap/1.0/";

        Element el = doc.createElementNS(namespace, name);
        el.setTextContent(content);
        return el;
    }

    private boolean isXmpMetadataDifferent(byte[] existingBytes, byte[] newBytes) {
        if (existingBytes == null || newBytes == null) {
            return true;
        }
        try {
            DocumentBuilder builder = DocumentBuilderFactory.newInstance().newDocumentBuilder();
            Document doc1 = builder.parse(new ByteArrayInputStream(existingBytes));
            Document doc2 = builder.parse(new ByteArrayInputStream(newBytes));
            return !Objects.equals(
                    doc1.getDocumentElement().getTextContent().trim(),
                    doc2.getDocumentElement().getTextContent().trim());
        } catch (Exception e) {
            log.warn("XMP diff failed: {}", e.getMessage());
            return true;
        }
    }
}
