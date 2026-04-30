import axios from 'axios';
import type { AxiosInstance } from 'axios';
import { processContent } from './content';
import type * as t from './types';
import { createDefaultLogger } from './utils';

/**
 * Keenable Search API and Scraper
 *
 * The Keenable API (https://api.keenable.ai) exposes a unified search +
 * content-fetch surface:
 *  - POST /v1/search   -> organic results
 *  - GET  /v1/fetch    -> per-URL content extraction
 *  - POST /v1/feedback -> optional relevance feedback
 *
 * Authentication: X-API-Key header.
 *
 * OpenAPI spec: https://api.keenable.ai/openapi.json
 */

const DEFAULT_BASE_URL = 'https://api.keenable.ai';

interface KeenableSearchResultDTO {
  title: string;
  url: string;
  description: string;
  snippet?: string;
}

interface KeenableSearchResponseDTO {
  query: string;
  results: KeenableSearchResultDTO[];
}

interface KeenableFetchResponseDTO {
  url: string;
  title?: string;
  content: string;
  metadata?: Record<string, unknown>;
}

/** Scrape response shape for the Keenable scraper — mirrors the Serper/Firecrawl response style. */
export interface KeenableScrapeResponse {
  success: boolean;
  data?: {
    markdown?: string;
    title?: string;
    metadata?: t.ScrapeMetadata;
  };
  error?: string;
}

export interface KeenableConfig {
  keenableApiKey?: string;
  keenableApiUrl?: string;
  searchProfile?: string;
}

export interface KeenableScraperConfig extends KeenableConfig {
  timeout?: number;
  logger?: t.Logger;
}

function buildClient(
  apiKey: string,
  baseURL: string,
  timeout: number
): AxiosInstance {
  return axios.create({
    baseURL,
    timeout,
    headers: {
      'X-API-Key': apiKey,
      'Content-Type': 'application/json',
    },
  });
}

/**
 * Factory for the Keenable SearchAPI. Returns `{ getSources }` matching the
 * shape expected by `createSearchAPI`.
 */
export const createKeenableSearchAPI = (
  config: KeenableConfig = {}
): {
  getSources: (params: t.GetSourcesParams) => Promise<t.SearchResult>;
} => {
  const apiKey = config.keenableApiKey ?? process.env.KEENABLE_API_KEY ?? '';
  const baseURL = (
    config.keenableApiUrl ??
    process.env.KEENABLE_API_URL ??
    DEFAULT_BASE_URL
  ).replace(/\/+$/, '');
  const searchProfile = config.searchProfile;

  if (!apiKey) {
    throw new Error(
      'KEENABLE_API_KEY is required for the Keenable search provider'
    );
  }

  const client = buildClient(apiKey, baseURL, 15000);

  const getSources = async ({
    query,
    type,
  }: t.GetSourcesParams): Promise<t.SearchResult> => {
    if (!query.trim()) {
      return { success: false, error: 'Query cannot be empty' };
    }

    /**
     * Keenable currently returns only organic web results. Image/video/news
     * verticals are not provided by the public API; when the orchestrator
     * asks for a typed search we return an empty success so the tool keeps
     * functioning without spurious errors.
     */
    if (type && type !== 'search') {
      return {
        success: true,
        data: {
          organic: [],
          images: [],
          videos: [],
          news: [],
          topStories: [],
          relatedSearches: [],
        },
      };
    }

    try {
      const response = await client.post<KeenableSearchResponseDTO>(
        '/v1/search',
        {
          query,
          ...(searchProfile ? { profile: searchProfile } : {}),
        }
      );

      const organic = response.data.results.map((r, i) => ({
        position: i + 1,
        title: r.title,
        link: r.url,
        snippet: r.snippet ?? r.description,
      }));

      return {
        success: true,
        data: {
          organic,
          images: [],
          videos: [],
          news: [],
          topStories: [],
          relatedSearches: [],
        },
      };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return {
        success: false,
        error: `Keenable search failed: ${errorMessage}`,
      };
    }
  };

  return { getSources };
};

/**
 * Keenable scraper — uses GET /v1/fetch to retrieve extracted markdown-like
 * content for a URL. Implements the common BaseScraper surface so it is a
 * drop-in alternative to Firecrawl and the Serper scraper.
 */
export class KeenableScraper implements t.BaseScraper {
  private apiKey: string;
  private baseURL: string;
  private timeout: number;
  private logger: t.Logger;
  private client: AxiosInstance;

  constructor(config: KeenableScraperConfig = {}) {
    this.apiKey = config.keenableApiKey ?? process.env.KEENABLE_API_KEY ?? '';
    this.baseURL = (
      config.keenableApiUrl ??
      process.env.KEENABLE_API_URL ??
      DEFAULT_BASE_URL
    ).replace(/\/+$/, '');
    this.timeout = config.timeout ?? 15000;
    this.logger = config.logger || createDefaultLogger();
    this.client = buildClient(this.apiKey, this.baseURL, this.timeout);

    if (!this.apiKey) {
      this.logger.warn(
        'KEENABLE_API_KEY is not set. Keenable scraping will not work.'
      );
    }
  }

  async scrapeUrl(
    url: string,
    _options: Record<string, unknown> = {}
  ): Promise<[string, KeenableScrapeResponse]> {
    if (!this.apiKey) {
      return [url, { success: false, error: 'KEENABLE_API_KEY is not set' }];
    }
    try {
      const response = await this.client.get<KeenableFetchResponseDTO>(
        '/v1/fetch',
        {
          params: { url },
        }
      );
      const data = response.data;
      return [
        url,
        {
          success: true,
          data: {
            markdown: data.content,
            title: data.title,
            metadata: {
              sourceURL: data.url,
              url: data.url,
              title: data.title,
              ...(data.metadata ?? {}),
            } as t.ScrapeMetadata,
          },
        },
      ];
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return [
        url,
        { success: false, error: `Keenable fetch failed: ${errorMessage}` },
      ];
    }
  }

  extractContent(
    response: KeenableScrapeResponse
  ): [string, undefined | t.References] {
    const markdown = response.data?.markdown;
    if (!response.success || markdown == null || markdown === '') {
      return ['', undefined];
    }
    try {
      /**
       * Keenable returns pre-extracted text rather than full HTML; `processContent`
       * expects both HTML and markdown to extract references. When no HTML is
       * available we emit the markdown as-is and skip reference extraction.
       */
      const { markdown: cleaned, ...rest } = processContent('', markdown);
      return [cleaned, rest];
    } catch (error) {
      this.logger.error('Error processing Keenable content:', error);
      return [markdown, undefined];
    }
  }

  extractMetadata(response: KeenableScrapeResponse): t.ScrapeMetadata {
    if (!response.success || !response.data || !response.data.metadata) {
      return {};
    }
    return response.data.metadata;
  }
}

export const createKeenableScraper = (
  config: KeenableScraperConfig = {}
): KeenableScraper => new KeenableScraper(config);
