package com.booklore.tools.pdfwriter;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;

import java.time.LocalDate;
import java.util.LinkedHashSet;
import java.util.Set;

@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonIgnoreProperties(ignoreUnknown = true)
public class BookMetadataInput {

    private String title;
    private String subtitle;
    private String description;
    private String publisher;
    private String language;
    private LocalDate publishedDate;
    private Set<String> authors;
    private Set<String> categories;
    private String isbn10;
    private String isbn13;
    private String asin;
    private String googleId;
    private String goodreadsId;
    private String comicvineId;
    private String hardcoverId;
    private String seriesName;
    private Float seriesNumber;
    private Integer seriesTotal;

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getSubtitle() {
        return subtitle;
    }

    public void setSubtitle(String subtitle) {
        this.subtitle = subtitle;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public String getPublisher() {
        return publisher;
    }

    public void setPublisher(String publisher) {
        this.publisher = publisher;
    }

    public String getLanguage() {
        return language;
    }

    public void setLanguage(String language) {
        this.language = language;
    }

    public LocalDate getPublishedDate() {
        return publishedDate;
    }

    public void setPublishedDate(LocalDate publishedDate) {
        this.publishedDate = publishedDate;
    }

    public Set<String> getAuthors() {
        return authors;
    }

    public void setAuthors(Set<String> authors) {
        if (authors == null) {
            this.authors = null;
        } else {
            this.authors = new LinkedHashSet<>(authors);
        }
    }

    public Set<String> getCategories() {
        return categories;
    }

    public void setCategories(Set<String> categories) {
        if (categories == null) {
            this.categories = null;
        } else {
            this.categories = new LinkedHashSet<>(categories);
        }
    }

    public String getIsbn10() {
        return isbn10;
    }

    public void setIsbn10(String isbn10) {
        this.isbn10 = isbn10;
    }

    public String getIsbn13() {
        return isbn13;
    }

    public void setIsbn13(String isbn13) {
        this.isbn13 = isbn13;
    }

    public String getAsin() {
        return asin;
    }

    public void setAsin(String asin) {
        this.asin = asin;
    }

    public String getGoogleId() {
        return googleId;
    }

    public void setGoogleId(String googleId) {
        this.googleId = googleId;
    }

    public String getGoodreadsId() {
        return goodreadsId;
    }

    public void setGoodreadsId(String goodreadsId) {
        this.goodreadsId = goodreadsId;
    }

    public String getComicvineId() {
        return comicvineId;
    }

    public void setComicvineId(String comicvineId) {
        this.comicvineId = comicvineId;
    }

    public String getHardcoverId() {
        return hardcoverId;
    }

    public void setHardcoverId(String hardcoverId) {
        this.hardcoverId = hardcoverId;
    }

    public String getSeriesName() {
        return seriesName;
    }

    public void setSeriesName(String seriesName) {
        this.seriesName = seriesName;
    }

    public Float getSeriesNumber() {
        return seriesNumber;
    }

    public void setSeriesNumber(Float seriesNumber) {
        this.seriesNumber = seriesNumber;
    }

    public Integer getSeriesTotal() {
        return seriesTotal;
    }

    public void setSeriesTotal(Integer seriesTotal) {
        this.seriesTotal = seriesTotal;
    }
}
