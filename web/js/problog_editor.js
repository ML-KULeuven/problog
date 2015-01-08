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
  hostname: 'http://localhost:5100',
  main_editor_url: '',
  editors: [],
  selector: '.problog-editor',
};

/** Initialize the header and all .problog-editor divs.
  *
  * Settings:
  * - resize: Boolean (false)
 **/
problog.initialize = function(settings) {

  var resize = false;

  if (settings !== undefined) {
    if (settings.resize === true) {
      resize = true;
    }
  }

  $(problog.selector).each(function(i,el) {
    problog.initDiv($(el), resize)
  });

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

  el.css('width', '84ex');

  // Init theory
  var intr = undefined;
  if (el.children('.interpretations').length > 0) {
    intr = el.children('.interpretations').html();
    el.children('.interpretations').remove();
  }
  var theory = el.html();
  el.html('');

  // DOM structure
  var new_id = 'problog-editor-n'+problog.editors.length;

  var problog_container = $('<div id="'+new_id+'"></div>').appendTo(el);
  var editor_container = $('<div class="editor-container" style="width:100%;height:300px;"></div>').appendTo(problog_container);
  editor_container.html(theory);
  if (intr !== undefined) {
    var editor_container_intr = $('<div class="editor-container-intr" style="width:100%;height:300px;"></div>').appendTo(problog_container);
    editor_container_intr.html(intr);
  }

  var buttons = $('<form class="text-center form-inline"></form>').appendTo(problog_container);
  var btn_group = $('<div class="btn-group" role="group"></div>').appendTo(buttons);
  var eval_btn = $('<input class="btn btn-default" type="button" value="Evaluate"/>').appendTo(btn_group);

  var result_panel = $('<div class="panel panel-default"><div class="panel-heading"><span class="panel-title">Result</span></div></div>').appendTo(problog_container);
  var result_panel_body = $('<div class="panel-body" class="result-final"></div>').appendTo(result_panel);

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
    eval_btn.val('Learn');
  }

  // Init buttons
  eval_btn.click(function() {
    var btn_txt = eval_btn.val();
    eval_btn.attr('disabled', 'disabled')
    eval_btn.val('processing...');
    var cur_model = editor.getSession().getValue();
    var cur_model_hash = undefined;
    if (CryptoJS !== undefined) {
      cur_model_hash = CryptoJS.MD5(cur_model);
    }
    var cur_intr = editor_intr.getSession().getValue();

    var url = problog.hostname + '/api/inference';
    var data = {'model': cur_model};
    if (intr !== undefined) {
      url = problog.hostname + '/api/learning';
      data['examples'] = cur_intr;
    }

    $.ajax({
      url: url, 
      dataType: 'jsonp',
      data: data,

    }).done( function(data) {

      if (intr !== undefined) {
        var meta = data;
        data = data.weights;
      }

      var result = $('<tbody>');
      for (var k in data) {
        var p = data[k];
        result.append($('<tr>')
              .append($('<td>').text(k))
              .append($('<td>').text(p)));
      }
      result = $('<table>', {'class': 'table'})
       .append($('<thead>')
        .append($('<tr>')
         .append($('<th>').text(intr?'Fact':'Query'))
         .append($('<th>').text('Probability'))
        )
       ).append(result);

      result_panel_body.html(result);

      if (intr !== undefined) {
        var meta_str = "<p><strong>Stats</strong>:";
        for (var k in meta) {
          if (k !== 'weights') {
            meta_str += " "+k+"="+meta[k];
          }
        }
        meta_str += "</p>";
        $(meta_str).appendTo(result_panel_body);
      } else {
        if (cur_model_hash) {
          $('<a href="'+problog.main_editor_url+'#hash='+cur_model_hash+'">Link to model</a>').appendTo(result_panel_body);
        }
      }

      eval_btn.removeAttr('disabled');
      eval_btn.val(btn_txt);

    }).fail( function(jqXHR, textStatus, errorThrown) {
      var result = $('<div>', {'class' : 'alert alert-danger'}).text( jqXHR.responseText);
      result_panel_body.html(result);

      if (cur_model_hash) {
        $('<a href="'+problog.main_editor_url+'#hash='+cur_model_hash+'">Link to model</a>').appendTo(result_panel_body);
      }

      eval_btn.removeAttr('disabled');
      eval_btn.val(btn_txt);
    });

  });

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

  problog.editors.push({editor: editor, resize: resizeEditor, id: new_id});

};

/** Load model from hash in editor **/
problog.fetchModel = function(hash, editor) {

  // Look at url if hash not given
  hash = problog.getHashFromUrl();
  if (hash == '') {
    return;
  }

  // Take first editor if not given
  if (editor === undefined) {
    if (problog.editors.length == 0 || problog.editors[0].editor == undefined) {
      return;
    }
    editor = problog.editors[0].editor;
  }

  $.ajax({
    url: problog.hostname + '/api/model', 
    dataType: 'jsonp',
    data: {'hash': hash},

  }).done( function(data) {
    editor.setValue(data.model,-1);

  }).fail( function(jqXHR, textStatus, errorThrown) {
    editor.setValue(jqXHR.responseText);
  });

};

problog.getHashFromUrl = function() {
  var hash = window.location.hash;
  hashidx = hash.indexOf("hash=");
  if (hashidx > 0) {
    hash = hash.substr(hashidx+5, hashidx+5+32);
  } else {
    hash = '';
  }
  return hash;
};


