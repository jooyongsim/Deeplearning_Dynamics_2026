/* Layout exporter for the HTML deck → PPTX build.
 *
 * Loaded by the deck only when opened with ?export (after KaTeX has rendered).
 * Walks every slide and writes one JSON document into <pre id="export">, which
 * `render_deck.sh export` captures with headless Chrome (--dump-dom).
 *
 * Item kinds (all geometry in slide px, 1280x720):
 *   text   {x,y,w,h, align, lh, runs:[{t|m|br, size, bold, italic, color, font, display}]}
 *   rect   {x,y,w,h, fill, stroke, sw, radius}        diagram blocks, bars, lines
 *   oval   {x,y,w,h, fill, stroke, sw}
 *   tri    {pts:[[x,y]x3], fill}
 *   line   {x1,y1,x2,y2, color, sw, dash, head}       SVG lines / paths (polyline via pts)
 *   img    {x,y,w,h, src}
 */
(function () {
  function waitMath(cb) {
    if (document.body.getAttribute('data-math') === 'done') return cb();
    setTimeout(function () { waitMath(cb); }, 100);
  }

  function rgbHex(c) {
    var m = /rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/.exec(c || '');
    if (!m) return null;
    if (m[4] !== undefined && parseFloat(m[4]) === 0) return null;
    return [m[1], m[2], m[3]].map(function (v) { return (+v).toString(16).padStart(2, '0'); }).join('').toUpperCase();
  }

  function fontKind(ff) {
    ff = ff.toLowerCase();
    if (ff.indexOf('consolas') >= 0) return 'mono';
    if (ff.indexOf('pretendard') >= 0) return 'display';
    if (ff.indexOf('noto sans kr') === 0 || ff.indexOf('"noto sans kr"') === 0) return 'hangul';
    return 'core';
  }

  function runStyle(el) {
    var cs = getComputedStyle(el);
    return {
      size: parseFloat(cs.fontSize),
      bold: parseInt(cs.fontWeight, 10) >= 600,
      italic: cs.fontStyle === 'italic',
      color: rgbHex(cs.color),
      font: fontKind(cs.fontFamily)
    };
  }

  function texOf(katexEl) {
    var a = katexEl.querySelector('annotation[encoding="application/x-tex"]');
    return a ? a.textContent.trim() : '';
  }

  // Serialise the inline content of a block into runs. `skip(el)` excludes
  // nested blocks that are exported on their own (e.g. nested <ul>).
  function runsOf(block, skip) {
    var runs = [];
    function walk(node) {
      if (node.nodeType === 3) {
        var keep = !!node.parentElement.closest('.code');     // code listings keep their indentation
        var t = keep ? node.textContent.replace(/\n/g, '') : node.textContent.replace(/\s+/g, ' ');
        if (t) {
          var r = runStyle(node.parentElement); r.t = t; runs.push(r);
        }
        return;
      }
      if (node.nodeType !== 1) return;
      if (skip && skip(node)) return;
      if (node.classList.contains('katex')) {
        var r = runStyle(node);
        r.m = texOf(node);
        r.display = !!node.closest('.katex-display');
        runs.push(r);
        return;
      }
      if (node.tagName === 'BR') { runs.push({br: true}); return; }
      // block-level children inside an inline flow (e.g. two display equations
      // in one .eq) become line breaks
      var disp = getComputedStyle(node).display;
      var isBlock = (disp === 'block' || disp === 'flex') && node !== block;
      if (isBlock && runs.length && !runs[runs.length - 1].br) runs.push({br: true});
      for (var c = node.firstChild; c; c = c.nextSibling) walk(c);
    }
    walk(block);
    // trim leading/trailing whitespace runs
    while (runs.length && runs[0].t !== undefined && !runs[0].t.trim()) runs.shift();
    while (runs.length && runs[runs.length - 1].br) runs.pop();
    var code = block.classList && block.classList.contains('code');
    if (runs.length && runs[0].t && !code) runs[0].t = runs[0].t.replace(/^\s+/, '');
    var last = runs[runs.length - 1];
    if (last && last.t && !code) last.t = last.t.replace(/\s+$/, '');
    return runs;
  }

  // Number of rendered line boxes: text-node rects plus whole KaTeX boxes, grouped by
  // vertical overlap (a KaTeX box spans its sub/superscripts, so it stays one line).
  function lineCount(el) {
    var rects = [];
    var walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    for (var n = walker.nextNode(); n; n = walker.nextNode()) {
      if (n.parentElement.closest('.katex') || !n.textContent.trim()) continue;
      var r = document.createRange(); r.selectNodeContents(n);
      Array.prototype.forEach.call(r.getClientRects(), function (q) { if (q.width > 0) rects.push(q); });
    }
    el.querySelectorAll('.katex').forEach(function (k) { rects.push(k.getBoundingClientRect()); });
    rects.sort(function (a, b) { return a.top - b.top; });
    var lines = 0, bottom = -1e9;
    rects.forEach(function (q) {
      if (q.top >= bottom - 3) { lines++; bottom = q.bottom; } else bottom = Math.max(bottom, q.bottom);
    });
    return Math.max(lines, 1);
  }

  function box(el, origin) {
    var r = el.getBoundingClientRect();
    return {x: r.left - origin.left, y: r.top - origin.top, w: r.width, h: r.height};
  }

  // Blocks exported as one text item each
  var TEXT_SEL = [
    'p.lead', 'p.def', 'p.sub-h', 'p.cap', 'p.code', 'div.eq', '.eqlabel', '.flab', '.tfig .src', '.dlabel',
    '.blk > span', '.op-oval', '.op-txt', 'th', 'td',
    '.eqgrid .n', '.eqgrid .k', '.eqgrid .m',
    '.cover h1', '.cover .meta > div', '.cover .sub'
  ].join(',');

  function exportSlide(s, idx) {
    var o = s.getBoundingClientRect();
    var items = [];
    var root = s;

    // ---- shapes ----------------------------------------------------------
    root.querySelectorAll('.blk').forEach(function (b) {
      var cs = getComputedStyle(b), bx = box(b, o);
      items.push(Object.assign({kind: 'rect', fill: rgbHex(cs.backgroundColor), stroke: rgbHex(cs.borderTopColor),
                                sw: parseFloat(cs.borderTopWidth), radius: parseFloat(cs.borderTopLeftRadius)}, bx));
    });
    root.querySelectorAll('.op-oval').forEach(function (b) {
      var cs = getComputedStyle(b);
      items.push(Object.assign({kind: 'oval', fill: null, stroke: rgbHex(cs.borderTopColor),
                                sw: parseFloat(cs.borderTopWidth)}, box(b, o)));
    });
    root.querySelectorAll('.vline, .hline, .arw').forEach(function (b) {
      items.push(Object.assign({kind: 'rect', fill: '111111', stroke: null, sw: 0, radius: 0}, box(b, o)));
      if (b.classList.contains('arw')) {        // ::after head: 17 wide, 26 tall, overlapping by 1px
        var bx = box(b, o), cy = bx.y + bx.h / 2;
        items.push({kind: 'tri', fill: '111111',
                    pts: [[bx.x + bx.w - 1, cy - 13], [bx.x + bx.w + 16, cy], [bx.x + bx.w - 1, cy + 13]]});
      }
    });
    root.querySelectorAll('.tri-d').forEach(function (b) {
      var bx = box(b, o);  // border triangle: 26 wide, 17 tall, apex down
      items.push({kind: 'tri', fill: '111111',
                  pts: [[bx.x, bx.y], [bx.x + 26, bx.y], [bx.x + 13, bx.y + 17]]});
    });

    // generic borders → lines (tables, eqbox keyline, equation grid rows)
    root.querySelectorAll('.content *, .content').forEach(function (el) {
      if (el.closest('.blk') || el.classList.contains('op-oval') || el.closest('svg') || el.closest('.katex')) return;
      var cs = getComputedStyle(el), bx = box(el, o);
      var bb = parseFloat(cs.borderBottomWidth), bl = parseFloat(cs.borderLeftWidth);
      if (bb > 0 && cs.borderBottomStyle !== 'none')
        items.push({kind: 'rect', x: bx.x, y: bx.y + bx.h - bb, w: bx.w, h: bb,
                    fill: rgbHex(cs.borderBottomColor), stroke: null, sw: 0, radius: 0});
      if (bl > 0 && cs.borderLeftStyle !== 'none')
        items.push({kind: 'rect', x: bx.x, y: bx.y, w: bl, h: bx.h,
                    fill: rgbHex(cs.borderLeftColor), stroke: null, sw: 0, radius: 0});
    });

    // ---- SVG figures -------------------------------------------------------
    root.querySelectorAll('.fig svg').forEach(function (svg) {
      var sr = svg.getBoundingClientRect();
      var ox = sr.left - o.left, oy = sr.top - o.top;
      svg.querySelectorAll('line, circle, path').forEach(function (el) {
        if (el.closest('marker')) return;
        var cs = getComputedStyle(el);
        var stroke = rgbHex(cs.stroke), sw = parseFloat(cs.strokeWidth) || 0;
        var dash = cs.strokeDasharray && cs.strokeDasharray !== 'none';
        var head = !!el.getAttribute('marker-end');
        if (el.tagName === 'line') {
          items.push({kind: 'line', svg: true, color: stroke, sw: sw, dash: dash, head: head,
                      pts: [[ox + +el.getAttribute('x1'), oy + +el.getAttribute('y1')],
                            [ox + +el.getAttribute('x2'), oy + +el.getAttribute('y2')]]});
        } else if (el.tagName === 'circle') {
          var cx = +el.getAttribute('cx'), cy = +el.getAttribute('cy'), r = +el.getAttribute('r');
          items.push({kind: 'oval', svg: true, x: ox + cx - r, y: oy + cy - r, w: 2 * r, h: 2 * r,
                      fill: rgbHex(cs.fill), stroke: stroke, sw: sw});
        } else {
          var L = el.getTotalLength(), n = Math.max(12, Math.round(L / 6)), pts = [];
          for (var k = 0; k <= n; k++) {
            var p = el.getPointAtLength(L * k / n);
            pts.push([ox + p.x, oy + p.y]);
          }
          items.push({kind: 'line', svg: true, color: stroke, sw: sw, dash: dash, head: head, pts: pts});
        }
      });
    });

    // ---- images --------------------------------------------------------------
    root.querySelectorAll('.tfig img, .fig img').forEach(function (im) {
      items.push(Object.assign({kind: 'img', src: im.getAttribute('src')}, box(im, o)));
    });

    // ---- bullets (li): marker + text, nested lists exported separately ------
    root.querySelectorAll('ul.lec li').forEach(function (li) {
      var cs = getComputedStyle(li), bx = box(li, o);
      var marker = getComputedStyle(li, '::before').content.replace(/"/g, '');
      var st = runStyle(li);
      var pad = parseFloat(cs.paddingLeft);
      // first-line box: measure with a range over the li's own inline content
      var rng = document.createRange();
      var firstKid = li.firstChild, sub = li.querySelector(':scope > ul');
      rng.setStartBefore(firstKid);
      if (sub) rng.setEndBefore(sub); else rng.setEndAfter(li.lastChild);
      var rr = rng.getBoundingClientRect();
      items.push(Object.assign({kind: 'text', align: 'left', lh: parseFloat(cs.lineHeight),
                                runs: [Object.assign({}, st, {t: marker})]},
                               {x: bx.x + 2, y: rr.top - o.top, w: pad, h: parseFloat(cs.lineHeight)}));
      items.push({kind: 'text', align: 'left', lh: parseFloat(cs.lineHeight),
                  x: bx.x + pad, y: rr.top - o.top, w: bx.w - pad, h: rr.height,
                  runs: runsOf(li, function (n) { return n.tagName === 'UL'; })});
    });

    // ---- text blocks -----------------------------------------------------------
    root.querySelectorAll(TEXT_SEL).forEach(function (el) {
      // skip blocks nested in another exported block (e.g. .blk span inside th? none) and hidden
      var parentBlock = el.parentElement.closest(TEXT_SEL);
      if (parentBlock && !el.classList.contains('eqlabel')) return;
      var runs = runsOf(el, function (n) { return n !== el && n.classList && n.classList.contains('eqlabel'); });
      if (!runs.length) return;
      var cs = getComputedStyle(el);
      var lh = parseFloat(cs.lineHeight);
      if (isNaN(lh)) lh = parseFloat(cs.fontSize) * 1.2;
      // use the content box (padding excluded) — table cells carry right padding
      var bx = box(el, o);
      var pl = parseFloat(cs.paddingLeft) || 0, pr = parseFloat(cs.paddingRight) || 0,
          pt = parseFloat(cs.paddingTop) || 0, pb = parseFloat(cs.paddingBottom) || 0;
      bx = {x: bx.x + pl, y: bx.y + pt, w: bx.w - pl - pr, h: bx.h - pt - pb};
      var lab = el.querySelector(':scope > .eqlabel');
      // an equation line with a label: the math starts where the first formula starts
      if (lab) {
        var k = el.querySelector('.katex');
        var kx = k.getBoundingClientRect().left - o.left;
        bx = {x: kx, y: bx.y, w: bx.x + bx.w - kx, h: bx.h};
      }
      var align = cs.textAlign;
      if (el.matches('.blk > span, .op-oval, .cover h1')) align = 'center';
      if (align === 'start') align = 'left';
      var valign = (el.tagName === 'TD' || el.tagName === 'TH') ? cs.verticalAlign : 'top';
      if (el.classList.contains('eqlabel')) {            // label: centre on its equation row
        var row = box(el.parentElement, o);
        bx = {x: bx.x, y: row.y, w: bx.w, h: row.h}; valign = 'middle';
      }
      if (lab) valign = 'middle';
      items.push(Object.assign({kind: 'text', align: align, valign: valign, lh: lh, runs: runs,
                                nlines: lineCount(el),
                                cls: el.classList[0] || el.tagName.toLowerCase()}, bx));
    });

    return {index: idx + 1, cover: s.classList.contains('cover'), title: s.dataset.title || '', items: items};
  }

  waitMath(function () {
    setTimeout(function () {
      var deck = document.getElementById('deck');
      deck.style.transform = 'none';
      var slides = Array.prototype.slice.call(deck.querySelectorAll('.slide'));
      var out = [];
      slides.forEach(function (s, i) {
        var was = s.style.display;
        slides.forEach(function (t) { t.style.display = 'none'; });
        s.style.display = 'block';
        out.push(exportSlide(s, i));
        s.style.display = was;
      });
      var pre = document.createElement('pre');
      pre.id = 'export';
      pre.textContent = JSON.stringify({slides: out});
      document.body.appendChild(pre);
    }, 800);
  });
})();
