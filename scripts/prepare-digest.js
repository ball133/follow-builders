#!/usr/bin/env node

// ============================================================================
// Follow Builders — Prepare Digest
// ============================================================================
// Gathers everything the LLM needs to produce a digest:
// - Fetches the central feeds (tweets + podcasts)
// - Fetches the latest prompts from GitHub
// - Reads the user's config (language, delivery method)
// - Outputs a single JSON blob to stdout
//
// The LLM's ONLY job is to read this JSON, remix the content, and output
// the digest text. Everything else is handled here deterministically.
//
// Usage: node prepare-digest.js
// Output: JSON to stdout
// ============================================================================

import { readFile, mkdir, writeFile } from 'fs/promises';
import { existsSync } from 'fs';
import { join, resolve } from 'path';
import { homedir } from 'os';
import { config as loadEnv } from 'dotenv';

// -- Constants ---------------------------------------------------------------

const USER_DIR = join(homedir(), '.follow-builders');
const CONFIG_PATH = join(USER_DIR, 'config.json');
const ENV_PATH = join(USER_DIR, '.env');

const FEED_X_URL = 'https://raw.githubusercontent.com/zarazhangrui/follow-builders/main/feed-x.json';
const FEED_PODCASTS_URL = 'https://raw.githubusercontent.com/zarazhangrui/follow-builders/main/feed-podcasts.json';
const FEED_BLOGS_URL = 'https://raw.githubusercontent.com/zarazhangrui/follow-builders/main/feed-blogs.json';

const PROMPTS_BASE = 'https://raw.githubusercontent.com/zarazhangrui/follow-builders/main/prompts';
const PROMPT_FILES = [
  'summarize-podcast.md',
  'summarize-tweets.md',
  'summarize-blogs.md',
  'digest-intro.md',
  'translate.md'
];

// -- Fetch helpers -----------------------------------------------------------

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) return null;
  return res.json();
}

async function fetchText(url) {
  const res = await fetch(url);
  if (!res.ok) return null;
  return res.text();
}

async function readJsonFile(filePath) {
  const raw = await readFile(filePath, 'utf-8');
  return JSON.parse(raw.replace(/^\uFEFF/, ''));
}

function truncateText(text, maxChars) {
  if (!text) return text;
  if (text.length <= maxChars) return text;
  return text.slice(0, maxChars) + '\n...[truncated]...';
}

function getArgValue(args, flag) {
  const idx = args.indexOf(flag);
  if (idx === -1) return null;
  const value = args[idx + 1];
  if (!value || value.startsWith('--')) return null;
  return value;
}

function dateYmdInTimeZone(timeZone, date = new Date()) {
  const fmt = new Intl.DateTimeFormat('en-CA', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit'
  });
  return fmt.format(date);
}

function buildLlmInput(output) {
  return {
    generatedAt: output.generatedAt,
    config: output.config,
    stats: output.stats,
    podcasts: (output.podcasts || []).map(p => ({
      ...p,
      transcript: truncateText(p.transcript, 12000)
    })),
    x: (output.x || []).map(a => ({
      ...a,
      tweets: (a.tweets || []).map(t => ({
        ...t,
        text: truncateText(t.text, 1500)
      }))
    })),
    blogs: (output.blogs || []).map(b => ({
      ...b,
      content: truncateText(b.content, 12000)
    })),
    errors: output.errors
  };
}

function buildMdTemplate(dateYmd, periodLabel, builderNames) {
  const uniqueBuilders = Array.from(new Set((builderNames || []).filter(Boolean)));
  const xSections =
    uniqueBuilders.length > 0
      ? uniqueBuilders
          .map(name =>
            [
              `### ${name}`,
              '',
              '- **Point:**',
              '- **My take:**',
              '',
            ].join('\n'),
          )
          .join('\n')
      : [
          '### Highlights',
          '',
          '- **Point:**',
          '- **My take:**',
          '',
        ].join('\n');

  return [
    '---',
    `title: "AI Builders Digest — ${dateYmd}"`,
    `date: ${dateYmd}`,
    'source: "AI Builders Digest"',
    `period: ${periodLabel}`,
    'tags:',
    '  - ai/infra',
    '  - ai/agents',
    '  - thesis/investing',
    '  - source/newsletter',
    ...(uniqueBuilders.length > 0
      ? ['people:', ...uniqueBuilders.map(n => `  - ${n}`)]
      : ['people: []']),
    'themes: []',
    'tickers: []',
    'status: processed',
    '---',
    '',
    `# AI Builders Digest — ${dateYmd}`,
    '',
    '## Snapshot',
    '',
    '- **Market theme:**',
    '- **Infra / tooling:**',
    '- **Agents / products:**',
    '- **Personal action for this week:**',
    '',
    '---',
    '',
    '## X / Twitter Highlights',
    '',
    xSections.trimEnd(),
    '',
    '---',
    '',
    '## Podcast / Long‑form',
    '',
    '---',
    '',
    '## Investment & Trading Angles',
    '',
    '---',
    '',
    '## Build / Research Backlog',
    '',
    '---',
    '',
    '## Links & Metadata',
    '',
    '### MOCs',
    '',
    '- [[AI – Agents MOC]]',
    '- [[AI – Infra MOC]]',
    '- [[AI – Investing MOC]]'
  ].join('\n');
}

async function generateDigestWithDeepseek(output) {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) throw new Error('DEEPSEEK_API_KEY not found in .env');

  const baseUrl = (process.env.DEEPSEEK_BASE_URL || 'https://api.deepseek.com').replace(/\/$/, '');
  const model = process.env.DEEPSEEK_MODEL || 'deepseek-v4-flash';

  const system = [
    'Output only the final digest in Markdown. Do not output JSON.',
    'Follow these instructions exactly.',
    '--- digest_intro ---',
    output.prompts?.digest_intro || '',
    '--- summarize_podcast ---',
    output.prompts?.summarize_podcast || '',
    '--- summarize_tweets ---',
    output.prompts?.summarize_tweets || '',
    '--- summarize_blogs ---',
    output.prompts?.summarize_blogs || '',
    '--- translate ---',
    output.prompts?.translate || ''
  ].join('\n\n');

  const user = JSON.stringify(buildLlmInput(output));

  const res = await fetch(`${baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: user }
      ],
      temperature: 0.2,
      stream: false
    }),
    signal: AbortSignal.timeout(120000)
  });

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`DeepSeek API error (${res.status}): ${text || res.statusText}`);
  }

  const data = await res.json();
  const content = data?.choices?.[0]?.message?.content;
  if (!content) throw new Error('DeepSeek API returned empty content');
  return content.trim();
}

async function generateMdNoteWithDeepseek(output, dateYmd, periodLabel) {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) throw new Error('DEEPSEEK_API_KEY not found in .env');

  const baseUrl = (process.env.DEEPSEEK_BASE_URL || 'https://api.deepseek.com').replace(/\/$/, '');
  const model = process.env.DEEPSEEK_MODEL || 'deepseek-v4-flash';

  const builders = (output.x || [])
    .map(a => a.name || a.username || a.handle)
    .filter(Boolean);
  const builderList = Array.from(new Set(builders)).slice(0, 30);

  const system = [
    'You are a careful summarizer and analyst writing a Markdown note for Obsidian.',
    'Output must be valid Markdown only. Do not output JSON.',
    'The output MUST match this structure exactly: YAML frontmatter, then headings/sections.',
    'Do not add extra top-level sections. Fill the existing fields/sections.',
    'YAML rules: use proper indentation. Arrays must be YAML lists (dash items).',
    'Frontmatter requirements:',
    `- title must be exactly: AI Builders Digest — ${dateYmd}`,
    `- date must be exactly: ${dateYmd}`,
    `- period must be exactly: ${periodLabel}`,
    "- tags must include: ai/infra, ai/agents, thesis/investing, source/newsletter",
    '- people should include the builder names you reference in the X section (use display names, not @handles).',
    '- themes must be 3-7 kebab-case items that describe the day (e.g., token-budgeting, agents, local-models).',
    '- tickers must be an array (can be empty).',
    "- status must be: processed",
    '',
    'Content rules:',
    '- Under each builder heading, fill Point + My take.',
    '- Include links: under each builder section add a "Links:" line and 1-3 Markdown links to the most relevant tweets for that builder (use the tweet URLs).',
    "- Do not remove any headings. You may leave a builder section blank only if there is truly no content, but still keep the heading.",
    '- Do not invent facts, metrics, or tickers. If unclear, write a cautious note.',
    '',
    'Use this template verbatim and fill it in:',
    '```markdown',
    buildMdTemplate(dateYmd, periodLabel, builderList),
    '```',
    '',
    'Builders available today (use as people candidates; include only those you actually mention):',
    builderList.length > 0 ? builderList.map(n => `- ${n}`).join('\n') : '(none)',
    '',
    'Additional guidance (summaries):',
    '--- summarize_podcast ---',
    output.prompts?.summarize_podcast || '',
    '--- summarize_tweets ---',
    output.prompts?.summarize_tweets || '',
    '--- summarize_blogs ---',
    output.prompts?.summarize_blogs || ''
  ].join('\n');

  const user = JSON.stringify(buildLlmInput(output));

  const res = await fetch(`${baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: user }
      ],
      temperature: 0.2,
      stream: false
    }),
    signal: AbortSignal.timeout(120000)
  });

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`DeepSeek API error (${res.status}): ${text || res.statusText}`);
  }

  const data = await res.json();
  const content = data?.choices?.[0]?.message?.content;
  if (!content) throw new Error('DeepSeek API returned empty content');
  return content.trim();
}

// -- Main --------------------------------------------------------------------

async function main() {
  const errors = [];

  loadEnv({ path: ENV_PATH });

  // 1. Read user config
  let config = {
    language: 'en',
    frequency: 'daily',
    delivery: { method: 'stdout' }
  };
  if (existsSync(CONFIG_PATH)) {
    try {
      config = await readJsonFile(CONFIG_PATH);
    } catch (err) {
      errors.push(`Could not read config: ${err.message}`);
    }
  }

  // 2. Fetch all three feeds
  const [feedX, feedPodcasts, feedBlogs] = await Promise.all([
    fetchJSON(FEED_X_URL),
    fetchJSON(FEED_PODCASTS_URL),
    fetchJSON(FEED_BLOGS_URL)
  ]);

  if (!feedX) errors.push('Could not fetch tweet feed');
  if (!feedPodcasts) errors.push('Could not fetch podcast feed');
  if (!feedBlogs) errors.push('Could not fetch blog feed');

  // 3. Load prompts with priority: user custom > remote (GitHub) > local default
  //
  // If the user has a custom prompt at ~/.follow-builders/prompts/<file>,
  // use that (they personalized it — don't overwrite with remote updates).
  // Otherwise, fetch the latest from GitHub so they get central improvements.
  // If GitHub is unreachable, fall back to the local copy shipped with the skill.
  const prompts = {};
  const scriptDir = decodeURIComponent(new URL('.', import.meta.url).pathname);
  const localPromptsDir = join(scriptDir, '..', 'prompts');
  const userPromptsDir = join(USER_DIR, 'prompts');

  for (const filename of PROMPT_FILES) {
    const key = filename.replace('.md', '').replace(/-/g, '_');
    const userPath = join(userPromptsDir, filename);
    const localPath = join(localPromptsDir, filename);

    // Priority 1: user's custom prompt (they personalized it)
    if (existsSync(userPath)) {
      prompts[key] = await readFile(userPath, 'utf-8');
      continue;
    }

    // Priority 2: latest from GitHub (central updates)
    const remote = await fetchText(`${PROMPTS_BASE}/${filename}`);
    if (remote) {
      prompts[key] = remote;
      continue;
    }

    // Priority 3: local copy shipped with the skill
    if (existsSync(localPath)) {
      prompts[key] = await readFile(localPath, 'utf-8');
    } else {
      errors.push(`Could not load prompt: ${filename}`);
    }
  }

  // 4. Build the output — everything the LLM needs in one blob
  const output = {
    status: 'ok',
    generatedAt: new Date().toISOString(),

    // User preferences
    config: {
      language: config.language || 'en',
      frequency: config.frequency || 'daily',
      delivery: config.delivery || { method: 'stdout' }
    },

    // Content to remix
    podcasts: feedPodcasts?.podcasts || [],
    x: feedX?.x || [],
    blogs: feedBlogs?.blogs || [],

    // Stats for the LLM to reference
    stats: {
      podcastEpisodes: feedPodcasts?.podcasts?.length || 0,
      xBuilders: feedX?.x?.length || 0,
      totalTweets: (feedX?.x || []).reduce((sum, a) => sum + a.tweets.length, 0),
      blogPosts: feedBlogs?.blogs?.length || 0,
      feedGeneratedAt: feedX?.generatedAt || feedPodcasts?.generatedAt || feedBlogs?.generatedAt || null
    },

    // Prompts — the LLM reads these and follows the instructions
    prompts,

    // Non-fatal errors
    errors: errors.length > 0 ? errors : undefined
  };

  const args = process.argv.slice(2);
  const outPathArg = getArgValue(args, '--out');
  const outPath = outPathArg ? resolve(process.cwd(), outPathArg) : null;
  const timeZone = config.timezone || process.env.DIGEST_TIMEZONE || 'UTC';
  const dateYmd = dateYmdInTimeZone(timeZone);
  const periodLabel =
    (config.frequency || 'daily').toLowerCase() === 'weekly' ? 'Weekly' : 'Daily';

  if (args.includes('--render-md')) {
    const md = await generateMdNoteWithDeepseek(output, dateYmd, periodLabel);
    if (outPath) {
      await writeFile(outPath, md + '\n', 'utf-8');
    }
    console.log(md);
    return;
  }

  if (args.includes('--render')) {
    const digest = await generateDigestWithDeepseek(output);
    if (outPath) {
      await writeFile(outPath, digest + '\n', 'utf-8');
    }
    console.log(digest);
    return;
  }

  console.log(JSON.stringify(output, null, 2));
}

main().catch(err => {
  console.error(JSON.stringify({
    status: 'error',
    message: err.message
  }));
  process.exit(1);
});
