// Build: cargo build -p kadmium-condenser-app --locked
// Setup: npm ci --prefix crates/kadmium-gateway/tests/ui
// Run: node --test crates/kadmium-condenser/tests/local_compute.test.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const os = require('node:os');
const { spawn } = require('node:child_process');
const { chromium } = require(path.resolve('crates/kadmium-gateway/tests/ui/node_modules/playwright'));

test('model preparation, local and Kadmium Cloud routing, quality tradeoffs, recovery and distributed jobs fit desktop and mobile', { timeout: 120000 }, async () => {
  const cwd = fs.mkdtempSync(path.join(os.tmpdir(), 'condenser-compute-ui-'));
  const source = path.join(cwd, 'source'); fs.mkdirSync(source);
  fs.writeFileSync(path.join(source, 'config.json'), JSON.stringify({ model_type: 'llama', max_position_embeddings: 8192 }));
  fs.writeFileSync(path.join(source, 'tokenizer.json'), '{}');
  const header = Buffer.from(JSON.stringify({ 'model.weight': { dtype: 'F32', shape: [2], data_offsets: [0, 8] } }));
  const length = Buffer.alloc(8); length.writeBigUInt64LE(BigInt(header.length));
  fs.writeFileSync(path.join(source, 'model.safetensors'), Buffer.concat([length, header, Buffer.from([0,0,128,63,1,0,128,63])]));
  const server = spawn(path.resolve('target/debug/kadmium-condenser' + (process.platform === 'win32' ? '.exe' : '')), ['serve', '--port', '0'], { cwd, windowsHide: true, env: { ...process.env, KADMIUM_MODEL_LIBRARY: path.join(cwd, 'library') } });
  let browser;
  try {
    const base = await new Promise((resolve,reject) => {
      server.once('error',reject);
      server.stdout.on('data',chunk => { const match=String(chunk).match(/http:\/\/127\.0\.0\.1:\d+/); if(match) resolve(match[0]); });
      server.once('exit',code=>reject(new Error('server exited: '+code)));
    });
    browser=await chromium.launch({channel:process.env.KADMIUM_TEST_BROWSER || (process.platform==='win32'?'msedge':undefined),headless:true});
    const page=await browser.newPage({viewport:{width:1440,height:1000},reducedMotion:'reduce'});
    const errors=[];page.on('pageerror',error=>errors.push(error.message));
    await page.goto(base);
    await page.locator('#open-model-library').click();
    await page.locator('#checkpoint-path').fill(source);
    assert.equal(await page.locator('#checkpoint-working-directory').inputValue(), '');
    const work = path.join(cwd, 'normalization-work'); fs.mkdirSync(work);
    await page.locator('#checkpoint-working-directory').fill(path.join(cwd, 'missing-work'));
    await page.locator('#checkpoint-import').click();
    await page.waitForFunction(()=>document.querySelector('#checkpoint-status').textContent.includes('Cannot open checkpoint working directory'));
    await page.locator('#checkpoint-working-directory').fill(work);
    await page.locator('#checkpoint-import').click();
    await page.waitForFunction(()=>document.querySelector('#checkpoint-status').textContent.includes('Normalized 1 tensors'));
    const id=await page.locator('#checkpoint-select').inputValue(); assert.match(id,/^[a-f0-9]{64}$/);
    assert.deepEqual(fs.readdirSync(work), []);
    assert.match(await page.locator('#checkpoint-facts').innerText(),/F32 · unchanged/);
    assert.match(await page.locator('#checkpoint-status').textContent(),/verified transport chunks/);
    assert.equal(await page.locator('#step-prepare').getAttribute('data-state'),'done');
    assert.equal(await page.locator('#step-run').getAttribute('data-state'),'current');
    assert.match(await page.locator('#checkpoint-context-hint').textContent(),/Maximum — 8,192 tokens/);
    await page.locator('#checkpoint-context').fill('2048');
    assert.match(await page.locator('#checkpoint-context-hint').textContent(),/Reduced to 2,048 tokens \(25% of 8,192\).*forgets/);
    await page.locator('#checkpoint-context').fill('');
    const before=await page.locator('#header-model').textContent();
    await page.route('**/api/v1/checkpoints/load',route=>route.fulfill({status:502,contentType:'application/json',body:JSON.stringify({error:'Backend changed F32 to F16; task route was not changed'})}));
    await page.locator('#checkpoint-load').click();
    await page.getByText('Backend changed F32 to F16; task route was not changed').waitFor();
    assert.equal(await page.locator('#header-model').textContent(),before);
    assert.equal(await page.locator('#checkpoint-use').isDisabled(),true);
    await page.unroute('**/api/v1/checkpoints/load');
    await page.route('**/api/v1/checkpoints/load',route=>route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({checkpoint:{id,name:'source'},provider:{kind:'ollama',model:'kadmium-'+id+':latest',base_url:'http://127.0.0.1:11434'},execution:'Local Ollama; source precision preserved.'})}));
    await page.locator('#checkpoint-load').click();
    await page.waitForFunction(()=>!document.querySelector('#checkpoint-use').disabled);
    assert.equal(await page.locator('#header-model').textContent(),before);
    await page.locator('#checkpoint-use').click();
    assert.match(await page.locator('#header-model').textContent(),new RegExp(id));
    assert.equal(await page.locator('#view-harness').isVisible(),true);
    await page.locator('#open-model-library').click();
    await page.locator('#checkpoint-backend').fill('http://127.0.0.1:11435');
    assert.equal(await page.locator('#checkpoint-use').isDisabled(),true);

    // Kadmium Cloud: send, follow progress, then route the task to the remote model.
    await page.locator('label:has(#destination-cloud)').click();
    assert.equal(await page.locator('#destination-cloud-panel').isVisible(),true);
    assert.equal(await page.locator('#cloud-endpoint').inputValue(),'https://api.kadmium.ai');
    let polls=0;
    await page.route('**/api/v1/checkpoints/publish',route=>{
      if(route.request().method()==='POST'){
        const body=route.request().postDataJSON();
        assert.deepEqual(body,{checkpoint_id:id,endpoint:'https://api.kadmium.ai',api_key:'cloud-key',context_length:null});
        return route.fulfill({json:{state:'uploading',checkpoint_id:id,total_bytes:2000,sent_bytes:500,skipped_bytes:0,message:'Sending verified 16 MiB chunks'}});
      }
      polls+=1;
      return route.fulfill({json:polls<2?{state:'uploading',checkpoint_id:id,total_bytes:2000,sent_bytes:1000,skipped_bytes:0,message:'Sending verified 16 MiB chunks'}
        :{state:'ready',checkpoint_id:id,total_bytes:2000,sent_bytes:2000,skipped_bytes:0,message:'Verified and loaded on Kadmium',result:{checkpoint:{id,name:'source',context_length:8192},context_length:8192,provider:{kind:'kadmium',model:'kadmium-'+id+':latest',base_url:'https://api.kadmium.ai'}}}});
    });
    await page.locator('#cloud-key').fill('cloud-key');
    await page.locator('#cloud-send').click();
    await page.waitForFunction(()=>!document.querySelector('#cloud-use').disabled);
    assert.match(await page.locator('#cloud-status').textContent(),/Ready on Kadmium with 8,192 tokens/);
    assert.equal(await page.locator('#cloud-progress').getAttribute('aria-valuenow'),'100');
    await page.locator('#cloud-use').click();
    assert.match(await page.locator('#header-provider').textContent(),/Kadmium/);
    await page.unroute('**/api/v1/checkpoints/publish');

    // Quality controls start at maximum and explain each reduction.
    await page.locator('#provider-button').click();
    assert.match(await page.locator('#quality-summary').textContent(),/Maximum quality/);
    await page.locator('#max-tokens').fill('4096');
    assert.match(await page.locator('#quality-summary').textContent(),/Reduced quality — responses capped at 4,096 tokens/);
    assert.match(await page.locator('#max-tokens-hint').textContent(),/stop mid-way/);
    await page.locator('#quality-maximum').click();
    assert.match(await page.locator('#quality-summary').textContent(),/Maximum quality/);
    await page.locator('#cancel-settings').click();
    await page.locator('.rail-nav[data-view="harness"]').click();
    await page.locator('#open-compute').click();
    await page.route('**/api/v1/compute/jobs',route=>{
      const body=route.request().postDataJSON();assert.equal(body.job.kind,'matrix_multiply');
      return route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({success:true,distributed:true,units_total:2,units_validated:2,nodes_used:['a','b'],wall_ms:3.25,traces:[],result:{values:[58,64,139,154]}})});
    });
    await page.locator('#compute-job-run').click();
    await page.getByText('Completed · 2/2 units · 2 workers · 3.25 ms').waitFor();
    assert.match(await page.locator('#compute-job-result').textContent(),/154/);
    await page.reload();await page.locator('#open-model-library').click();
    await page.waitForFunction(id=>Array.from(document.querySelector('#checkpoint-select').options).some(option=>option.value===id),id);
    await page.locator('#checkpoint-select').selectOption(id);
    fs.mkdirSync(path.resolve('target/checkpoint-smoke'),{recursive:true});
    for(const width of [1440,768,390]) {
      await page.setViewportSize({width,height:900});
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
      await page.waitForFunction(()=>document.querySelector('#run-card').getBoundingClientRect().right<=innerWidth);
      const bounds=await page.locator('#run-card').boundingBox();await page.screenshot({path:path.resolve('target/checkpoint-smoke/models-'+width+'.png')});assert.ok(bounds.x>=0&&bounds.x+bounds.width<=width, JSON.stringify({width,bounds}));
    }
    await page.setViewportSize({width:390,height:844});
    await page.evaluate(()=>document.body.classList.add('context-closed'));
    assert.equal(await page.locator('#session-rail').getAttribute('aria-hidden'),'true');
    assert.equal(await page.locator('#session-rail').evaluate(el=>el.inert),true);
    assert.equal(await page.locator('#toggle-sessions').getAttribute('aria-expanded'),'false');
    await page.locator('#toggle-sessions').focus();
    const closedTabTrail=[];
    for(let i=0;i<6;i++){
      await page.keyboard.press('Tab');
      closedTabTrail.push(await page.evaluate(()=>({id:document.activeElement.id,inRail:Boolean(document.activeElement.closest('#session-rail'))})));
    }
    assert.equal(closedTabTrail.some(item=>item.inRail),false);
    await page.locator('#toggle-sessions').click();
    await page.waitForFunction(()=>document.querySelector('#session-rail').contains(document.activeElement));
    assert.equal(await page.locator('#session-rail').getAttribute('aria-hidden'),'false');
    assert.equal(await page.locator('#session-rail').evaluate(el=>el.inert),false);
    assert.equal(await page.locator('#toggle-sessions').getAttribute('aria-expanded'),'true');
    for(let i=0;i<10;i++){
      await page.keyboard.press('Tab');
      assert.equal(await page.evaluate(()=>Boolean(document.activeElement.closest('#session-rail'))),true);
    }
    await page.keyboard.press('Escape');
    await page.waitForFunction(()=>document.activeElement.id==='toggle-sessions');
    assert.equal(await page.locator('#session-rail').evaluate(el=>el.inert),true);
    await page.locator('#toggle-sessions').click();
    await page.locator('#session-backdrop').click({position:{x:340,y:400}});
    await page.waitForFunction(()=>document.activeElement.id==='toggle-sessions');
    assert.equal(await page.locator('#session-rail').evaluate(el=>el.inert),true);
    await page.locator('#toggle-sessions').click();
    await page.locator('#session-rail [data-session]').first().click();
    await page.waitForFunction(()=>document.querySelector('#session-rail').inert);
    assert.equal(await page.evaluate(()=>Boolean(document.activeElement.closest('#session-rail'))),false);
    await page.setViewportSize({width:1440,height:1000});
    await page.waitForFunction(()=>!document.querySelector('#session-rail').inert);
    assert.equal(await page.locator('#session-rail').getAttribute('aria-hidden'),'false');
    await page.screenshot({path:path.resolve('target/checkpoint-smoke/condenser-compute.png')});
    assert.deepEqual(errors,[]);
  } finally {
    if(browser) await browser.close();
    server.kill(); await new Promise(resolve=>server.exitCode===null?server.once('exit',resolve):resolve());
    fs.rmSync(cwd,{recursive:true,force:true,maxRetries:5,retryDelay:100});
  }
});
