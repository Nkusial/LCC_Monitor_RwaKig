import { createServer } from 'node:http'
import { readFile } from 'node:fs/promises'
import { existsSync, mkdirSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const root = path.resolve(__dirname, '..', '..')
const appDir = path.join(root, 'docs', 'app')
const assetsDir = path.join(root, 'docs', 'assets')
const port = 5193

const mimeTypes = {
  '.css': 'text/css',
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
}

function serveDocsApp() {
  return createServer(async (request, response) => {
    try {
      const requestUrl = new URL(request.url ?? '/', `http://127.0.0.1:${port}`)
      const relativePath = requestUrl.pathname === '/' ? 'index.html' : requestUrl.pathname.slice(1)
      const filePath = path.normalize(path.join(appDir, relativePath))

      if (!filePath.startsWith(appDir)) {
        response.writeHead(403)
        response.end('Forbidden')
        return
      }

      const body = await readFile(filePath)
      response.writeHead(200, {
        'Content-Type': mimeTypes[path.extname(filePath)] ?? 'application/octet-stream',
      })
      response.end(body)
    } catch {
      const fallback = await readFile(path.join(appDir, 'index.html'))
      response.writeHead(200, { 'Content-Type': 'text/html' })
      response.end(fallback)
    }
  })
}

async function launchBrowser() {
  for (const channel of ['chrome', 'msedge']) {
    try {
      return await chromium.launch({ channel })
    } catch {
      // Try the next locally installed browser channel.
    }
  }

  return chromium.launch()
}

async function capture(page, url, outputPath, viewport) {
  await page.setViewportSize(viewport)
  await page.goto(url, { waitUntil: 'networkidle' })
  await page.locator('.maplibregl-canvas').waitFor({ timeout: 15000 })
  await page.locator('.validation-panel').waitFor({ timeout: 15000 })
  await page.waitForTimeout(1000)
  await page.screenshot({ path: outputPath, fullPage: false })
}

if (!existsSync(appDir)) {
  throw new Error(`Build docs app first: ${appDir} does not exist`)
}

mkdirSync(assetsDir, { recursive: true })

const server = serveDocsApp()
await new Promise((resolve) => server.listen(port, '127.0.0.1', resolve))

const browser = await launchBrowser()
const page = await browser.newPage()

try {
  const baseUrl = `http://127.0.0.1:${port}`
  await capture(
    page,
    `${baseUrl}/?layer=unsupervised_clusters_best`,
    path.join(assetsDir, '08_validation_overlay.png'),
    { width: 1600, height: 1200 },
  )

  await capture(
    page,
    `${baseUrl}/?layer=unsupervised_clusters_best&validationWorkflow=1`,
    path.join(assetsDir, '09_validation_workflow.png'),
    { width: 1600, height: 2200 },
  )
} finally {
  await browser.close()
  server.close()
}
