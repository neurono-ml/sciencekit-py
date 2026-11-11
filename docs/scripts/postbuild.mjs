#!/usr/bin/env node
// Post-process the mdBook output: set per-page canonical/Open Graph URLs and
// generate sitemap.xml so search engines discover every chapter.

import { readdir, readFile, stat, writeFile } from 'node:fs/promises';
import { basename, join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const BASE_URL = 'https://neurono-ml.github.io/sciencekit-py/';
const BOOK_DIR = fileURLToPath(new URL('../book', import.meta.url));
const SKIPPED_PAGES = new Set(['print.html', '404.html', 'toc.html']);

function isVerificationFile(fileName) {
  return /^google[0-9a-z]+\.html$/i.test(fileName) || /^bingsiteauth/i.test(fileName);
}

async function listHtmlFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await listHtmlFiles(path)));
    } else if (entry.name.endsWith('.html')) {
      files.push(path);
    }
  }
  return files;
}

function pageUrl(file) {
  const relativePath = relative(BOOK_DIR, file).split(sep).join('/');
  const canonicalPath = relativePath.replace(/(^|\/)index\.html$/, '$1');
  return BASE_URL + canonicalPath;
}

function escapeXml(value) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;');
}

function injectMetadata(html, url) {
  let updated = html;
  if (updated.includes('rel="canonical"')) {
    updated = updated.replace(
      /<link rel="canonical" href="[^"]*">/,
      `<link rel="canonical" href="${url}">`
    );
  } else {
    updated = updated.replace(
      '</head>',
      `        <link rel="canonical" href="${url}">\n    </head>`
    );
  }
  if (updated.includes('property="og:url"')) {
    updated = updated.replace(
      /<meta property="og:url" content="[^"]*">/,
      `<meta property="og:url" content="${url}">`
    );
  } else {
    updated = updated.replace(
      '</head>',
      `        <meta property="og:url" content="${url}">\n    </head>`
    );
  }
  return updated;
}

const files = (await listHtmlFiles(BOOK_DIR)).filter((file) => {
  const name = basename(file);
  return !SKIPPED_PAGES.has(name) && !isVerificationFile(name);
});

const urls = [];
for (const file of files) {
  const url = pageUrl(file);
  const html = await readFile(file, 'utf8');
  const updated = injectMetadata(html, url);
  if (updated !== html) await writeFile(file, updated);
  const { mtime } = await stat(file);
  urls.push({ loc: url, lastmod: mtime.toISOString().slice(0, 10) });
}

urls.sort((left, right) => left.loc.localeCompare(right.loc));

const sitemap = [
  '<?xml version="1.0" encoding="UTF-8"?>',
  '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
  ...urls.map(
    ({ loc, lastmod }) =>
      `  <url><loc>${escapeXml(loc)}</loc><lastmod>${lastmod}</lastmod></url>`
  ),
  '</urlset>',
  ''
].join('\n');

await writeFile(join(BOOK_DIR, 'sitemap.xml'), sitemap);
console.log(`postbuild: ${urls.length} pages indexed in sitemap.xml`);
