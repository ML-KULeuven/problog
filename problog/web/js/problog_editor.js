/* Problog v2.1 javascript interface
 *
 * Usage:
 * When calling `problog.initialize()`, all .problog-editor divs are replaced
 * by editors with the original contents of the div.
 * If the .problog-editor div contains a .interpretations div, it is considered
 * a learning task and a second editor is added for training data.
 *
 * Requires:
 * - http://jquery.com/
 * - http://getbootstrap.com/
 * - https://code.google.com/p/crypto-js/
 *
 * Copyright (c) 2015, Anton Dries, Wannes Meert, KU Leuven.
 * All rights reserved.
 */

var problog = {
  hostname: '//adams.cs.kuleuven.be/problog/api/',
  main_editor_url: 'https://dtai.cs.kuleuven.be/problog/editor.html',
  editors: [],
  selector: '.problog-editor',
  trackurl: false,
  resize: false,
};

/** Initialize the header and all .problog-editor divs.
 **/
problog.initialize = function(settings) {

  $.extend(problog, settings);

  $('head').append('<style type="text/css"> \
     .problog-edit-editor {height: 400px; width: 100%;} \
     .problog-editor {width:100%;} \
     .problog-editor-hash {float:right; margin-right:5px;} \
     .problog-edit-editor-small {height: 200px; width: 100%;} \
     .problog-result-sortable {cursor: pointer;} \
     .problog-result-sorted-asc::after {content: "\\025bc";} \
     .problog-result-sorted-desc::after {content: "\\025b2";} \
     .glyphicon-refresh-animate {-animation: spin .7s infinite linear; -webkit-animation: spin2 .7s infinite linear;} @-webkit-keyframes spin2 { from { -webkit-transform: rotate(0deg);} to { -webkit-transform: rotate(360deg); } @keyframes spin { from { transform: scale(1) rotate(0deg);} to { transform: scale(1) rotate(360deg);} \
     </style>');

  $(problog.selector).each(function(i,el) {
    problog.initDiv($(el), problog.resize)
  });

  if (problog.trackurl) {
    problog.trackUrlHash();
  }
};

/** Clear all .problog-editor divs. **/
problog.clear = function() {
  jquery(problog.selector).each(function(i,el) { $(el).html(''); });
};

/** Setup divs **/
problog.initDiv = function(el, resize) {

  if (resize === undefined) {
    resize = false;
  }

  if (el.data('width')) {
    el.css('width', el.data('width'));
  } else {
    el.css('width', '84ex');
  }

  // Init theory
  var intr = undefined;
  if (el.children('.interpretations').length > 0) {
    var intr_height = el.children('.interpretations').data('height');
    if (intr_height === undefined) {
      intr_height = '150px';
    }
    intr = el.children('.interpretations').html();
    if (intr[0] == '\n') {
      intr = intr.substr(1);
    }
    el.children('.interpretations').remove();
  }
  var theory = el.html();
  if (theory[0] == '\n') {
    theory = theory.substr(1);
  }
  el.html('');

  // DOM structure
  var new_id = 'problog-editor-n'+problog.editors.length;

  var problog_container = $('<div id="'+new_id+'"></div>').appendTo(el);
  var editor_container = $('<div class="problog-editor-container" style="width:100%;height:300px;"></div>').appendTo(problog_container);
  editor_container.html(theory);
  if (intr !== undefined) {
    var editor_container_intr = $('<div class="problog-editor-container-intr" style="width:100%;height:'+intr_height+';"></div>').appendTo(problog_container);
    editor_container_intr.html(intr);
  }

  var buttons = $('<form class="form-inline problog-editor-buttons"></form>').appendTo(problog_container);
  var btn_group = $('<div class="btn-group" role="group"></div>').appendTo(buttons);
  var eval_btn = $('<input class="btn btn-default" type="button" value="Evaluate"/>').appendTo(btn_group);

  //var result_panel = $('<div class="panel panel-default"><div class="panel-heading"><span class="panel-title">Result</span></div></div>').appendTo(problog_container);
  var result_panel = $('<div class="panel panel-default problog-editor-results"></div>').appendTo(problog_container);
  var result_panel_body = $('<div class="panel-body" class="result-final">Results ...</div>').appendTo(result_panel);

  var makeProgressBar = function(value) {
    return $('<div class="progress"><div class="progress-bar" role="progressbar" aria-valuenow="' + (value*100) + '" aria-valuemin="0" aria-valuemax="100" style="text-align:left; width: ' + (100*value) + '%;padding:3px;color:black;background-color:#9ac2f4;background-image: linear-gradient(to bottom,#d3f0ff 0,#8fccff 100%);">&nbsp'+ value + '</div></div>');
  }

  // Init ACE editor
  var editor = ace.edit(editor_container[0]);
  editor.getSession().setMode('ace/mode/prolog');
  editor.getSession().setUseWrapMode(true);
  editor.setShowInvisibles(true);

  if (intr !== undefined) {
    var editor_intr = ace.edit(editor_container_intr[0]);
    editor_intr.getSession().setMode('ace/mode/prolog');
    editor_intr.getSession().setUseWrapMode(true);
    editor_intr.setShowInvisibles(true);
    //eval_btn.val('Learn');
    var learn_btn = $('<input class="btn btn-default" type="button" value="Learn"/>').appendTo(btn_group);
  } else {
    var editor_intr = undefined;
    var learn_btn = undefined;
  }

  var start = function(btn, learn) {
    result_panel_body.html("...");
    var btn_txt = btn.val();
    btn.attr('disabled', 'disabled')
    btn.val('processing...');
    editor.getSession().clearAnnotations();
    var cur_model = editor.getSession().getValue();
    if (cur_model == '') {
      cur_model = '%%';
    }
    var cur_model_hash = undefined;
    if (CryptoJS !== undefined) {
      cur_model_hash = CryptoJS.MD5(cur_model);
    }

    var url = problog.hostname + 'inference';
    var data = {'model': cur_model};
    if (learn && intr !== undefined) {
      url = problog.hostname + 'learning';
      var cur_examples = editor_intr.getSession().getValue();
      if (cur_examples == '') {
        cur_examples = "%%";
      }
      data['examples'] = cur_examples;
      var cur_examples_hash = undefined;
      if (CryptoJS !== undefined) {
        cur_examples_hash = CryptoJS.MD5(cur_examples);
      }
    }

    $.ajax({
      url: url, 
      dataType: 'jsonp',
      data: data,
      success: function(data) {
        console.log(data);
        if (data.SUCCESS == true) {
          if (learn) {
            var facts = data.weights;
          } else {
            var facts = data.probs
          }

          var result = $('<tbody>');
          for (var k in facts) {
            var n = facts[k][0];
            var p = facts[k][1];
            var l = facts[k][2];
            var c = facts[k][3];
            result.append($('<tr>')
                  .append($('<td>').text(n))
                  .append($('<td>').text(l+':'+c))
                  .append($('<td>').append(makeProgressBar(p))));
          }

          result = $('<table>', {'class': 'table table-condensed'})
            .append($('<thead>')
             .append($('<tr>')
              .append($('<th class="problog-result-sortable" style="width:50%;">').text(learn?'Fact':'Query'))
              .append($('<th class="problog-result-sortable" style="width:10%;">').text('Location'))
              .append($('<th class="problog-result-sortable" style="width:40%;">').text('Probability'))
             )
            ).append(result);

          var table = result[0];
          problog.sortTable(table, 1, 0);
          result_panel_body.html(result);

          var col_th = $(table).children('thead').children('tr').children('th')
          col_th.eq(0).addClass('problog-result-sorted-asc');
          for (var i=0; i<col_th.length; i++) {
            col_th.eq(i).click((function(col_idx) { return function() {
              if ($(this).hasClass('problog-result-sorted-asc')) {
                $(this).removeClass('problog-result-sorted-asc');
                $(this).addClass('problog-result-sorted-desc');
                problog.sortTable(table, -1, col_idx);
              } else if ($(this).hasClass('problog-result-sorted-desc')) {
                $(this).removeClass('problog-result-sorted-desc');
                $(this).addClass('problog-result-sorted-asc');
                problog.sortTable(table, 1, col_idx);
              } else {
                col_th.removeClass('problog-result-sorted-asc');
                col_th.removeClass('problog-result-sorted-desc');
                $(this).addClass('problog-result-sorted-asc');
                problog.sortTable(table, 1, col_idx);
              }
            };})(i));
          }

          if (learn) {
            var meta_str = "<p><strong>Stats</strong>:";
            var sep = " ";
            for (var k in data) {
              if (k !== 'weights' && k !== 'probs' && k !== 'SUCCESS' && k !== 'url') {
                meta_str += sep+k+"="+data[k];
                sep = ", ";
              }
            }
            meta_str += "</p>";
            $(meta_str).appendTo(result_panel_body);
          }

        } else {
          p = data.err;
          var msg = p.message;
          if (msg == undefined) msg = p;

          if (p.location) {
            var row = p.location[1];
            var col = p.location[2];
          } else {
            var row = p.lineno;
            var col = p.colno;
          }
          var result = $('<div>', { 'class': 'alert alert-danger' } ).text(msg);
          if (row !== undefined) {
            var editor = ace.edit(editor_container[0]);
            editor.getSession().setAnnotations([{ row: row-1, column: col, text: p.message, type: 'error'}]);	   
          }
          result_panel_body.html(result);

        }

        if (cur_model_hash) {
          var cur_url = problog.main_editor_url+'#hash='+cur_model_hash;
          if (cur_examples_hash) {
            cur_url += '&ehash='+cur_examples_hash;
          }
          var clipboardBtn = $('<a class="problog-editor-hash" href="'+cur_url+'">Link to model&nbsp;<span class="problog-editor-hash glyphicon glyphicon-share" aria-hidden="true"></span></a>').appendTo(result_panel_body);
          clipboardBtn.attr('title','Click to copy url to clipboard.');
          clipboardBtn.click(function(e) {
            e.preventDefault();
            window.prompt("Copy to clipboard: Ctrl/Cmd+C, Enter", cur_url);
          });
        }

        btn.removeAttr('disabled');
        btn.val(btn_txt);




      },

      error: function(jqXHR, textStatus, errorThrown) {
        //console.log("Problog request failed");
        // TODO: response text is not captured by jQuery?
        var text = "No (correct) response from server. ";
        if (jqXHR.responseText) {
          text += jqXHR.responseText;
        }
        var result = $('<div>', {'class' : 'alert alert-danger'}).text(text);
        result_panel_body.html(result);

        if (cur_model_hash) {
          var clipboardBtn = $('<a class="problog-editor-hash" href="'+problog.main_editor_url+'#hash='+cur_model_hash+'">Link to model&nbsp;<span class="problog-editor-hash glyphicon glyphicon-share" aria-hidden="true"></span></a>').appendTo(result_panel_body);
          clipboardBtn.attr('title','Click to copy url to clipboard.');
          clipboardBtn.click(function(e) {
            e.preventDefault();
            window.prompt("Copy to clipboard: Ctrl/Cmd+C, Enter", problog.main_editor_url+'#hash='+cur_model_hash);
          });
        }

        btn.removeAttr('disabled');
        btn.val(btn_txt);
      }
    });

  };

  // Init buttons
  eval_btn.click(function() {
    start(eval_btn, false);
  });
  if (learn_btn) {
    learn_btn.click(function() {
      start(learn_btn, true);
    });
  }

  // Auto Resize
  if (resize) {
    var resizeEditor = function() {
      // Resize the editor to the length of the contents
      var screenLength = editor.getSession().getScreenLength();
      if (screenLength < 5) { screenLength = 5 };
      var newHeight = screenLength * editor.renderer.lineHeight
                      + editor.renderer.scrollBar.getWidth();

      editor_container.height((newHeight+10).toString() + "px");

      // This call is required for the editor to fix all of
      // its inner structure for adapting to a change in size
      editor.resize();
    }
    resizeEditor();

    editor.getSession().on('change', function(e) {
      console.log(e.data.action);
      if (e.data.action == "insertText" || e.data.action == "removeLines") {
        console.log("Resize editor");
        resizeEditor();
      }
    });
  } else {
    var resizeEditor = undefined;
  }

  problog.editors.push({editor: editor, resize: resizeEditor, id: new_id, examples:editor_intr});

};

problog.trackUrlHash = function() {
  problog.fetchModel();
  $(window).on('hashchange', function() {
    problog.fetchModel();
  });
};

/** Load model from hash in editor **/
problog.fetchModel = function(hash, editor, ehash) {

  // Look at url if hash not given
  if (hash === undefined) {
    hash = problog.getHashFromUrl();
    if (hash == '') {
      return;
    }
  }

  if (ehash === undefined) {
    ehash = problog.getExamplesHashFromUrl();
  }

  // Take first editor if not given
  var editor_examples = undefined;
  if (editor === undefined) {
    if (problog.editors.length == 0 || problog.editors[0].editor == undefined) {
      return;
    }
    editor = problog.editors[0].editor;
    editor_examples = problog.editors[0].examples;
  }

  $.ajax({
    url: problog.hostname+'model', 
    dataType: 'jsonp',
    data: {'hash': hash},

  }).done( function(data) {
    if (data.SUCCESS == true) {
      editor.setValue(data.model,-1);
    } else {
      editor.setValue('% '+data.err,-1);
    }

  }).fail( function(jqXHR, textStatus, errorThrown) {
    editor.setValue(jqXHR.responseText);
  });

  if (ehash && editor_examples) {
    $.ajax({
      url: problog.hostname+'examples', 
      dataType: 'jsonp',
      data: {'ehash': ehash},

    }).done( function(data) {
      if (data.SUCCESS == true) {
        editor_examples.setValue(data.examples,-1);
      } else {
        editor_examples.setValue('% Failed loading: '+data.err,-1);
      }

    }).fail( function(jqXHR, textStatus, errorThrown) {
      editor_examples.setValue(jqXHR.responseText);
    });
  }

};


/** Extract 'hash' value from url
  * Example:
  *   http://dtai.cs.kuleuven.be/problog/editor.html#hash=xxxx&...
  */
problog.getHashFromUrl = function() {
  var hash = window.location.hash;
  hashidx = hash.indexOf("hash=");
  if (hashidx > 0) {
    hash = hash.substr(hashidx+5, 32);
    ampidx = hash.indexOf("&");
    if (ampidx > 0) {
      hash = hash.substring(0, ampidx);
    }
  } else {
    hash = '';
  }
  return hash;
};


/** Extract 'ehash' value from url
  * Example:
  *   http://dtai.cs.kuleuven.be/problog/editor.html#ehash=xxxx&...
  */
problog.getExamplesHashFromUrl = function() {
  var hash = window.location.hash;
  hashidx = hash.indexOf("ehash=");
  if (hashidx > 0) {
    hash = hash.substr(hashidx+6, 32);
    ampidx = hash.indexOf("&");
    if (ampidx > 0) {
      hash = hash.substring(0, ampidx);
    }
  } else {
    hash = '';
  }
  return hash;
};


/** Sort rows in table
  * 
  * Params:
  * table table element
  * dir direction of sort, 1=ascending, -1=descending
  * col number of column (td in tr)
  */
problog.sortTable = function(table, dir, col){
  //var rows = $('#mytable tbody  tr').get();
  var rows = $(table).find('tbody tr').get();
  rows.sort(function(a, b) {

    // get the text of col-th <td> of <tr> 
    var a = $(a).children('td').eq(col).text().toUpperCase();
    var b = $(b).children('td').eq(col).text().toUpperCase();
    //if(a < b) {
     //return -1*dir;
    //}
    //if(a > b) {
     //return 1*dir;
    //}
    //return 0;
    return dir*problog.naturalSort(a,b);
  });

  $.each(rows, function(index, row) {
    $(table).children('tbody').append(row);
  });
}

/*
 * Natural Sort algorithm for Javascript - Version 0.8.1 - Released under MIT license
 * Author: Jim Palmer (based on chunking idea from Dave Koelle)
 */
problog.naturalSort = function(a, b) {
  var re = /(^([+\-]?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?(?=\D|\s|$))|^0x[\da-fA-F]+$|\d+)/g,
    sre = /^\s+|\s+$/g,   // trim pre-post whitespace
    snre = /\s+/g,        // normalize all whitespace to single ' ' character
    dre = /(^([\w ]+,?[\w ]+)?[\w ]+,?[\w ]+\d+:\d+(:\d+)?[\w ]?|^\d{1,4}[\/\-]\d{1,4}[\/\-]\d{1,4}|^\w+, \w+ \d+, \d{4})/,
    hre = /^0x[0-9a-f]+$/i,
    ore = /^0/,
    i = function(s) {
      return (problog.naturalSort.insensitive && ('' + s).toLowerCase() || '' + s).replace(sre, '');
    },
    // convert all to strings strip whitespace
    x = i(a),
    y = i(b),
    // chunk/tokenize
    xN = x.replace(re, '\0$1\0').replace(/\0$/,'').replace(/^\0/,'').split('\0'),
    yN = y.replace(re, '\0$1\0').replace(/\0$/,'').replace(/^\0/,'').split('\0'),
    // numeric, hex or date detection
    xD = parseInt(x.match(hre), 16) || (xN.length !== 1 && Date.parse(x)),
    yD = parseInt(y.match(hre), 16) || xD && y.match(dre) && Date.parse(y) || null,
    normChunk = function(s, l) {
      // normalize spaces; find floats not starting with '0', string or 0 if not defined (Clint Priest)
      return (!s.match(ore) || l == 1) && parseFloat(s) || s.replace(snre, ' ').replace(sre, '') || 0;
    },
    oFxNcL, oFyNcL;
  // first try and sort Hex codes or Dates
  if (yD) {
    if (xD < yD) { return -1; }
    else if (xD > yD) { return 1; }
  }
  // natural sorting through split numeric strings and default strings
  for(var cLoc = 0, xNl = xN.length, yNl = yN.length, numS = Math.max(xNl, yNl); cLoc < numS; cLoc++) {
    oFxNcL = normChunk(xN[cLoc] || '', xNl);
    oFyNcL = normChunk(yN[cLoc] || '', yNl);
    // handle numeric vs string comparison - number < string - (Kyle Adams)
    if (isNaN(oFxNcL) !== isNaN(oFyNcL)) {
      return isNaN(oFxNcL) ? 1 : -1;
    }
    // if unicode use locale comparison
    if (/[^\x00-\x80]/.test(oFxNcL + oFyNcL) && oFxNcL.localeCompare) {
      var comp = oFxNcL.localeCompare(oFyNcL);
      return comp / Math.abs(comp);
    }
    if (oFxNcL < oFyNcL) { return -1; }
    else if (oFxNcL > oFyNcL) { return 1; }
  }
}


