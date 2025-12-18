package com.booklore.tools.pdf;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;

import java.time.LocalDate;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.Set;

/**
 * Minimal copy of the BookLore metadata DTO so the CLI can serialize
 * extractor results without depending on the rest of the application.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public final class BookMetadata {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper().registerModule(new JavaTimeModule());

    private final String title;
    private final String description;
    private final String publisher;
    private final String language;
    private final Set<String> authors;
    private final Set<String> categories;
    private final LocalDate publishedDate;
    private final String isbn10;
    private final String isbn13;
    private final String googleId;
    private final String asin;
    private final String goodreadsId;
    private final String comicvineId;
    private final String hardcoverId;
    private final String seriesName;
    private final Float seriesNumber;

    private BookMetadata(Builder builder) {
        this.title = builder.title;
        this.description = builder.description;
        this.publisher = builder.publisher;
        this.language = builder.language;
        this.authors = toUnmodifiable(builder.authors);
        this.categories = toUnmodifiable(builder.categories);
        this.publishedDate = builder.publishedDate;
        this.isbn10 = builder.isbn10;
        this.isbn13 = builder.isbn13;
        this.googleId = builder.googleId;
        this.asin = builder.asin;
        this.goodreadsId = builder.goodreadsId;
        this.comicvineId = builder.comicvineId;
        this.hardcoverId = builder.hardcoverId;
        this.seriesName = builder.seriesName;
        this.seriesNumber = builder.seriesNumber;
    }

    private static Set<String> toUnmodifiable(Set<String> values) {
        if (values == null || values.isEmpty()) {
            return Collections.emptySet();
        }
        return Collections.unmodifiableSet(new LinkedHashSet<>(values));
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getTitle() {
        return title;
    }

    public String getDescription() {
        return description;
    }

    public String getPublisher() {
        return publisher;
    }

    public String getLanguage() {
        return language;
    }

    public Set<String> getAuthors() {
        return authors;
    }

    public Set<String> getCategories() {
        return categories;
    }

    public LocalDate getPublishedDate() {
        return publishedDate;
    }

    public String getIsbn10() {
        return isbn10;
    }

    public String getIsbn13() {
        return isbn13;
    }

    public String getGoogleId() {
        return googleId;
    }

    public String getAsin() {
        return asin;
    }

    public String getGoodreadsId() {
        return goodreadsId;
    }

    public String getComicvineId() {
        return comicvineId;
    }

    public String getHardcoverId() {
        return hardcoverId;
    }

    public String getSeriesName() {
        return seriesName;
    }

    public Float getSeriesNumber() {
        return seriesNumber;
    }

    public String toPrettyJson() {
        try {
            return OBJECT_MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(this);
        } catch (JsonProcessingException e) {
            throw new IllegalStateException("Failed to serialize metadata", e);
        }
    }

    public static final class Builder {
        private String title;
        private String description;
        private String publisher;
        private String language;
        private Set<String> authors;
        private Set<String> categories;
        private LocalDate publishedDate;
        private String isbn10;
        private String isbn13;
        private String googleId;
        private String asin;
        private String goodreadsId;
        private String comicvineId;
        private String hardcoverId;
        private String seriesName;
        private Float seriesNumber;

        private Set<String> ensureMutable(Set<String> current) {
            if (current == null) {
                return new LinkedHashSet<>();
            }
            return current;
        }

        public Builder title(String title) {
            this.title = title;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder publisher(String publisher) {
            this.publisher = publisher;
            return this;
        }

        public Builder language(String language) {
            this.language = language;
            return this;
        }

        public Builder authors(Set<String> authors) {
            if (authors != null) {
                Set<String> holder = ensureMutable(this.authors);
                holder.clear();
                holder.addAll(authors);
                this.authors = holder;
            }
            return this;
        }

        public Builder categories(Set<String> categories) {
            if (categories != null) {
                Set<String> holder = ensureMutable(this.categories);
                holder.clear();
                holder.addAll(categories);
                this.categories = holder;
            }
            return this;
        }

        public Builder publishedDate(LocalDate publishedDate) {
            this.publishedDate = publishedDate;
            return this;
        }

        public Builder isbn10(String isbn10) {
            this.isbn10 = isbn10;
            return this;
        }

        public Builder isbn13(String isbn13) {
            this.isbn13 = isbn13;
            return this;
        }

        public Builder googleId(String googleId) {
            this.googleId = googleId;
            return this;
        }

        public Builder asin(String asin) {
            this.asin = asin;
            return this;
        }

        public Builder goodreadsId(String goodreadsId) {
            this.goodreadsId = goodreadsId;
            return this;
        }

        public Builder comicvineId(String comicvineId) {
            this.comicvineId = comicvineId;
            return this;
        }

        public Builder hardcoverId(String hardcoverId) {
            this.hardcoverId = hardcoverId;
            return this;
        }

        public Builder seriesName(String seriesName) {
            this.seriesName = seriesName;
            return this;
        }

        public Builder seriesNumber(Float seriesNumber) {
            this.seriesNumber = seriesNumber;
            return this;
        }

        public BookMetadata build() {
            return new BookMetadata(this);
        }
    }

    @Override
    public String toString() {
        return toPrettyJson();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof BookMetadata that)) return false;
        return Objects.equals(title, that.title)
                && Objects.equals(description, that.description)
                && Objects.equals(publisher, that.publisher)
                && Objects.equals(language, that.language)
                && Objects.equals(authors, that.authors)
                && Objects.equals(categories, that.categories)
                && Objects.equals(publishedDate, that.publishedDate)
                && Objects.equals(isbn10, that.isbn10)
                && Objects.equals(isbn13, that.isbn13)
                && Objects.equals(googleId, that.googleId)
                && Objects.equals(asin, that.asin)
                && Objects.equals(goodreadsId, that.goodreadsId)
                && Objects.equals(comicvineId, that.comicvineId)
                && Objects.equals(hardcoverId, that.hardcoverId)
                && Objects.equals(seriesName, that.seriesName)
                && Objects.equals(seriesNumber, that.seriesNumber);
    }

    @Override
    public int hashCode() {
        return Objects.hash(title, description, publisher, language, authors, categories, publishedDate,
                isbn10, isbn13, googleId, asin, goodreadsId, comicvineId, hardcoverId, seriesName, seriesNumber);
    }
}
