
import axios from 'axios';
import {
  createKeenableSearchAPI,
  createKeenableScraper,
  KeenableScraper,
} from './keenable';

jest.mock('axios');

const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('Keenable Search API', () => {
  const mockClient = {
    post: jest.fn(),
    get: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockedAxios.create = jest.fn(
      () => mockClient as unknown as ReturnType<typeof axios.create>
    );
  });

  test('getSources maps Keenable results into SearchResultData.organic', async () => {
    mockClient.post.mockResolvedValueOnce({
      data: {
        query: 'who is Andrey Styskin',
        results: [
          {
            title: 'Andrey Styskin — Yandex',
            url: 'https://example.com/andrey',
            description: 'Profile page',
            snippet: 'Former director of search quality at Yandex',
          },
          {
            title: 'Interview',
            url: 'https://example.com/interview',
            description: 'Long read',
          },
        ],
      },
    });

    const api = createKeenableSearchAPI({ keenableApiKey: 'key' });
    const result = await api.getSources({ query: 'who is Andrey Styskin' });

    expect(result.success).toBe(true);
    expect(result.data?.organic).toEqual([
      {
        position: 1,
        title: 'Andrey Styskin — Yandex',
        link: 'https://example.com/andrey',
        snippet: 'Former director of search quality at Yandex',
      },
      {
        position: 2,
        title: 'Interview',
        link: 'https://example.com/interview',
        snippet: 'Long read',
      },
    ]);
    expect(mockClient.post).toHaveBeenCalledWith('/v1/search', {
      query: 'who is Andrey Styskin',
    });
  });

  test('getSources rejects empty queries without calling the API', async () => {
    const api = createKeenableSearchAPI({ keenableApiKey: 'key' });
    const result = await api.getSources({ query: '   ' });
    expect(result).toEqual({ success: false, error: 'Query cannot be empty' });
    expect(mockClient.post).not.toHaveBeenCalled();
  });

  test('getSources returns empty verticals for non-web types (images/videos/news)', async () => {
    const api = createKeenableSearchAPI({ keenableApiKey: 'key' });
    const result = await api.getSources({ query: 'cats', type: 'images' });
    expect(result.success).toBe(true);
    expect(result.data?.organic).toEqual([]);
    expect(mockClient.post).not.toHaveBeenCalled();
  });

  test('missing API key throws on factory call', () => {
    const oldKey = process.env.KEENABLE_API_KEY;
    delete process.env.KEENABLE_API_KEY;
    expect(() => createKeenableSearchAPI({})).toThrow(
      /KEENABLE_API_KEY is required/
    );
    if (oldKey !== undefined) process.env.KEENABLE_API_KEY = oldKey;
  });
});

describe('Keenable Scraper', () => {
  const mockClient = {
    post: jest.fn(),
    get: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockedAxios.create = jest.fn(
      () => mockClient as unknown as ReturnType<typeof axios.create>
    );
  });

  test('scrapeUrl returns markdown content and metadata', async () => {
    mockClient.get.mockResolvedValueOnce({
      data: {
        url: 'https://example.com',
        title: 'Example Domain',
        content:
          'Example Domain. This domain is for use in illustrative examples.',
        metadata: { language: 'en' },
      },
    });

    const scraper = createKeenableScraper({ keenableApiKey: 'key' });
    const [url, response] = await scraper.scrapeUrl('https://example.com');

    expect(url).toBe('https://example.com');
    expect(response.success).toBe(true);
    expect(response.data?.markdown).toContain('Example Domain');
    expect(response.data?.title).toBe('Example Domain');
    expect(mockClient.get).toHaveBeenCalledWith('/v1/fetch', {
      params: { url: 'https://example.com' },
    });
  });

  test('scrapeUrl reports error when API throws', async () => {
    mockClient.get.mockRejectedValueOnce(new Error('boom'));
    const scraper = createKeenableScraper({ keenableApiKey: 'key' });
    const [, response] = await scraper.scrapeUrl('https://example.com');
    expect(response.success).toBe(false);
    expect(response.error).toContain('boom');
  });

  test('extractContent returns empty for unsuccessful responses', () => {
    const scraper = new KeenableScraper({ keenableApiKey: 'key' });
    expect(scraper.extractContent({ success: false })).toEqual(['', undefined]);
  });

  test('extractMetadata returns {} for unsuccessful responses', () => {
    const scraper = new KeenableScraper({ keenableApiKey: 'key' });
    expect(scraper.extractMetadata({ success: false })).toEqual({});
  });
});
