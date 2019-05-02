
var problog = {
  //hostname: 'https://verne.cs.kuleuven.be/problog/api/',
  hostname: 'https://verne.cs.kuleuven.be/problog/api/',
  main_editor_url: 'https://dtai.cs.kuleuven.be/problog/editor.html',
  editors: [],
  selector: '.problog-editor',
  trackurl: false,
  resize: false,
  tasks: [
                {
                    id: 'prob',
                    name: "Inference",
                    text: "Standard ProbLog inference task.",
                    action: 'Evaluate',
                    choices: [
                        {name:"-exact"},
                        {name:"SDD"},
                        // {name:"d-DNNF"},
                        // {name:"BDD"},
                        // {name:"-approximate"},
                        // {name:"forward"},
                        // {name:"k-best"},
                        // {name:"sample"}
                    ],
                    select: function(pbl) {},
                    deselect: function(pbl) {},
                    collectData: function(pbl) {
                        var solve_choice = 'default';
                        if (pbl.solve_choice !== undefined) {
                            solve_choice = pbl.task.choices[pbl.solve_choice].name;
                        }
                        var model = pbl.editor.getSession().getValue();

                        if (model) {
                            return {
                                'model': model
                                //'options': solve_choice
                            };
                        } else {
                            return undefined;
                        }
                    },
                    formatResult: function(pbl, data) {
                        var facts = data.probs
                        // Create table body
                        var result = $('<tbody>');
                        for (var k in facts) {
                            var n = facts[k][0];
                            var p = facts[k][1];
                            var l = facts[k][2];
                            var c = facts[k][3];
                            if (!isNaN(parseFloat(p))) {
                                p = problog.makeProgressBar(p);
                            }

                            result.append($('<tr>')
                                  .append($('<td>').text(n))
                                  .append($('<td>').text(l+':'+c))
                                  .append($('<td>').append(p)));
                        }
                        var result = problog.createTable(result, [['Query','50%'],['Location','10%'],['Probability','40%']]);
                        pbl.dom.results.html(result);
                    }
                },
                {
                    id: 'lfi',
                    name: "Learning",
                    text: "Learning from interpretations.",
                    action: 'Learn',
                    select: function(pbl) {
                        var editor = $('<div>').addClass("problog-edit-editor-small");
                        pbl.editor_data = ace.edit(editor[0]);
                        pbl.editor_data.getSession().setMode('ace/mode/problog');
                        pbl.editor_data.getSession().setUseWrapMode(true);
                        if (pbl.initial_data) {
                            pbl.editor_data.setValue(pbl.initial_data, -1);
                        }
                        pbl.editor_data.setShowInvisibles(true);
                        pbl.dom.edit_options.append($('<strong>').text('Examples (specified as evidence, separated by ---):'));
                        pbl.dom.edit_options.append(editor);
                    },
                    deselect: function(pbl) {
                        pbl.dom.edit_options.empty();
                    },
                    collectData: function(pbl) {
                        var model =  pbl.editor.getSession().getValue();
                        var examples = pbl.editor_data.getSession().getValue();
                        if (model && examples) {
                            return {
                                'model': model,
                                'examples': examples
                            };
                        } else {
                            return undefined;
                        }
                    },
                    formatResult: function(pbl, data) {
                        var facts = data.weights
                        var result = $('<tbody>');
                        for (var k in facts) {
                            var n = facts[k][0];
                            var p = facts[k][1];
                            var l = facts[k][2];
                            var c = facts[k][3];
                            result.append($('<tr>')
                                  .append($('<td>').text(n))
                                  .append($('<td>').text(l+':'+c))
                                  .append($('<td>').append(problog.makeProgressBar(p))));
                        }
                        var result = problog.createTable(result, [['Fact','50%'],['Location','10%'],['Probability','40%']]);
                        pbl.dom.results.html(result);

                        var model_str = "<strong>Model</strong>:<pre><code>" + data['model'] + "</code></pre>";

                        var meta_str = "<p><strong>Stats</strong>:";
                        var sep = " ";
                        for (var k in data) {
                            if (k !== 'weights' && k !== 'probs' && k != 'url' && k != 'model' && k !== 'SUCCESS') {
                                meta_str += sep+k+"="+data[k];
                                sep = ", ";
                            }
                        }
                        meta_str += "</p>";
                        $(model_str).appendTo(pbl.dom.results);
                        $(meta_str).appendTo(pbl.dom.results);
                    }
                },
                {
                    id: 'mpe', name: "MPE", select: function(pbl){}, deselect: function(pbl){},
                    collectData: function(pbl){
                        var model = pbl.editor.getSession().getValue();
                        if (model) {
                            return {
                                'model': model
                            };
                        } else {
                            return undefined;
                        }
                    },
                    formatResult: function(pbl, data) {
                        var facts = data.atoms
                        // Create table body
                        var result = $('<tbody>');
                        for (var k in facts) {
                            var n = facts[k][0];
                            var p = facts[k][1];
                            result.append($('<tr>')
                                  .append($('<td>').text(n))
                                  .append($('<td>').append(problog.makeProgressBar(p))));
                        }
                        var result = problog.createTable(result, [['Atom','50%'],['Value','50%']]);
                        pbl.dom.results.html(result);
                    }
                },
                    {
                    id: 'map',
                    name: "MAP",
                    action: 'Solve',
                    text: "Compute MAP.",
                    select: function(pbl){},
                    deselect: function(pbl){},
                    collectData: function(pbl){
                        var model = pbl.editor.getSession().getValue();
                        if (model) {
                            result = {'model': model};
                            if (pbl.solve_choice > 0) {
                                result['solve'] = pbl.task.choices[pbl.solve_choice].identifier;
                            }
                            return result;
                        } else {
                            return undefined;
                        }

                    },
                    formatResult: function(pbl, data) {
                        var facts = data.choices;

                        // Create table body
                        var result = $('<tbody>');
                        for (var k in facts) {
                            var n = facts[k][0];
                            var p = facts[k][1];
                            result.append($('<tr>')
                                  .append($('<td>').text(n))
                                  .append($('<td>').append(problog.makeProgressBar(p))));
                        }
                        var result = problog.createTable(result, [['Atom','50%'],['Value','50%']]);
                        pbl.dom.results.html(result);

                        var meta_str = "<p><strong>Score</strong>: ";
                        var sep = " ";
                        meta_str += data.score;
                        meta_str += "</p>";
                        $(meta_str).appendTo(pbl.dom.results);

                        var meta_str = "<p><strong>Stats</strong>:";
                        var sep = " ";
                        for (var k in data.stats) {
                            meta_str += sep+k+"="+data.stats[k];
                            sep = ", ";
                        }
                        meta_str += "</p>";
                        $(meta_str).appendTo(pbl.dom.results);
                      }
                    },
                    {
                    id: 'dt',
                    name: "DTProbLog",
                    action: 'Solve',
                    text: "Compute the optimal strategy.",
                    choices: [{'name': 'exact'}, {'name': 'local search', 'identifier': 'local'}],
                    select: function(pbl){},
                    deselect: function(pbl){},
                    collectData: function(pbl){
                        var model = pbl.editor.getSession().getValue();
                        if (model) {
                            result = {'model': model};
                            if (pbl.solve_choice > 0) {
                                result['solve'] = pbl.task.choices[pbl.solve_choice].identifier;
                            }
                            return result;
                        } else {
                            return undefined;
                        }

                    },
                    formatResult: function(pbl, data) {
                        var facts = data.choices;

                        // Create table body
                        var result = $('<tbody>');
                        for (var k in facts) {
                            var n = facts[k][0];
                            var p = facts[k][1];
                            result.append($('<tr>')
                                  .append($('<td>').text(n))
                                  .append($('<td>').append(problog.makeProgressBar(p))));
                        }
                        var result = problog.createTable(result, [['Atom','50%'],['Value','50%']]);
                        pbl.dom.results.html(result);

                        var meta_str = "<p><strong>Score</strong>: ";
                        var sep = " ";
                        meta_str += data.score;
                        meta_str += "</p>";
                        $(meta_str).appendTo(pbl.dom.results);

                        var meta_str = "<p><strong>Stats</strong>:";
                        var sep = " ";
                        for (var k in data.stats) {
                            meta_str += sep+k+"="+data.stats[k];
                            sep = ", ";
                        }
                        meta_str += "</p>";
                        $(meta_str).appendTo(pbl.dom.results);


                    }
                },
                // {
                //     id: 'ground',
                //     name: "Ground",
                //     action: 'Ground',
                //     select: function(pbl){},
                //     deselect: function(pbl){},
                //     collectData: function(pbl) {
                //         var model = pbl.editor.getSession().getValue();
                //         if (model) {
                //             return {
                //                 'model': model
                //             };
                //         } else {
                //             return undefined;
                //         }
                //     },
                //     formatResult: function(pbl, data) {
                //         var facts = data.probs
                //         // Create table body
                //         pbl.dom.results.html(data.result);
                //
                //     }
                // },
                {
                    id: 'sample',
                    name: "Sampling",
                    action: 'Sample',
                    text: "Generate samples from the model.",
                    select: function(pbl){},
                    deselect: function(pbl){},
                    collectData: function(pbl){
                        var model = pbl.editor.getSession().getValue();
                        if (model) {
                            return {
                                'model': model
                            };
                        } else {
                            return undefined;
                        }

                    },
                    formatResult: function(pbl, data) {
                        var facts = data.results[0]
                        // Create table body
                        var result = $('<tbody>');
                        for (var k in facts) {
                            var n = facts[k][0];
                            var p = facts[k][1];
                            result.append($('<tr>')
                                  .append($('<td>').text(n))
                                  .append($('<td>').append(problog.makeProgressBar(p))));
                        }
                        var result = problog.createTable(result, [['Atom','50%'],['Value','50%']]);
                        pbl.dom.results.html(result);
                    }

                },
                {
                    id: 'explain',
                    name: "Explain",
                    action: 'Explain',
                    text: "Explain how to obtain the probability.",
                    choices: [{'name': 'exact'}],
                    select: function(pbl){},
                    deselect: function(pbl){},
                    collectData: function(pbl){
                        var model = pbl.editor.getSession().getValue();
                        if (model) {
                            result = {'model': model};
                            if (pbl.solve_choice > 0) {
                                result['solve'] = pbl.task.choices[pbl.solve_choice].identifier;
                            }
                            return result;
                        } else {
                            return undefined;
                        }

                    },
                    formatResult: function(pbl, data) {
                        var program = data.program;
                        var proofs = data.proofs;
                        var facts = data.probabilities;

                        // Create table body
                        var result = $('<tbody>');
                        for (var k in facts) {
                            var n = facts[k][0];
                            var p = facts[k][1];
                            result.append($('<tr>')
                                  .append($('<td>').text(n))
                                  .append($('<td>').append(problog.makeProgressBar(p))));
                        }
                        var result = problog.createTable(result, [['Atom','50%'],['Value','50%']]);

                        var div_program = $('<div>').append($('<strong>').html('Transformed program')).append($('<pre>').html(program.join('<br>')));
                        var div_proofs = $('<div>').append($('<strong>').html('Mutually exclusive proofs')).append($('<pre>').html(proofs.join('<br>')));

                        var result = $('<div>').append(div_program).append(div_proofs).append(result);

                        pbl.dom.results.html(result);

                    }
                },
                {
                    id: 'english',
                    name: "Natural Language",
                    text: "Solve question in natural language.",
                    action: 'Solve',
                    choices: [
                        // {name:"-exact"},
                        // {name:"SDD"},
                        // {name:"d-DNNF"},
                        // {name:"BDD"},
                        // {name:"-approximate"},
                        // {name:"forward"},
                        // {name:"k-best"},
                        // {name:"sample"}
                    ],
                    select: function(pbl) {
                      pbl.natlang = false;
                      $('<h3>').text("Enter your question:").appendTo(pbl.dom.root.preface);
                      pbl.natlang_text = $('<textarea>', {'rows': 5}).addClass('form-control').appendTo(pbl.dom.root.preface);
                      var button = $('<button>').addClass("btn btn-primary pull-right")
                                                            .text('Solve')
                                                            .click(function() {
                                                              pbl.natlang = true;
                                                              pbl.solve();
                                                              pbl.natlang = false;
                                                             })
                                                            .appendTo(pbl.dom.root.preface);
                      $('<h3>').html('Or, enter your model:').appendTo(pbl.dom.root.preface);

                    },
                    deselect: function(pbl) {
                      pbl.dom.root.preface.empty();


                    },
                    collectData: function(pbl) {
                        if (pbl.natlang) {
                          var text = $(pbl.natlang_text).val();
                          if (text) {
                            return {'model': text, 'is_text': pbl.natlang};
                          } else {
                            return undefined;
                          }
                        } else {
                          var model = pbl.editor.getSession().getValue();
                          if (model) {
                              return {
                                'model': model,
                                'is_text': pbl.natlang
                                  //'options': solve_choice
                              };
                          } else {
                              return undefined;
                        }
                      }
                    },
                    formatResult: function(pbl, data) {
                        console.log(data);

//                        var facts = data.probs
//                        // Create table body
//                        var result = $('<tbody>');
//                        for (var k in facts) {
//                            var n = facts[k][0];
//                            var p = facts[k][1];
//                            var l = facts[k][2];
//                            var c = facts[k][3];
//                            if (!isNaN(parseFloat(p))) {
//                                p = problog.makeProgressBar(p);
//                            }
//
//                            result.append($('<tr>')
//                                  .append($('<td>').text(n))
//                                  .append($('<td>').text(l+':'+c))
//                                  .append($('<td>').append(p)));
//                        }
//                        var result = problog.createTable(result, [['Query','50%'],['Location','10%'],['Probability','40%']]);
                        var result = $('<div>');
                        result.append($('<span>').append($('<strong>').text('Solution: ')))
                                .append($('<span>', {'class': 'label label-default'}).text(data.solve_output));

                        pbl.editor.getSession().setValue(data.program);

                        console.log(result);
                        pbl.dom.results.html(result);
                    }
                },


                ]
}

problog.init = function(hostname) {
    if (hostname !== undefined) {
        problog.hostname = hostname;
    }

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
    $.each($(problog.selector), problog.init_editor);
}

problog.makeProgressBar = function(value) {
    return $('<div class="progress"><div class="progress-bar" role="progressbar" aria-valuenow="' + (value*100) + '" aria-valuemin="0" aria-valuemax="100" style="text-align:left; width: ' + (100*value) + '%;padding:3px;color:black;background-color:#9ac2f4;background-image: linear-gradient(to bottom,#d3f0ff 0,#8fccff 100%);">&nbsp'+ value + '</div></div>');
}

problog.init_editor = function(index, object) {

    // Create container object for editor settings and DOM.
    var pbl = { dom: {} };

    pbl.initial_data = '';
    var ex_data = $(object).find('.examples')[0];
    if (ex_data) {
        pbl.initial_data = $(ex_data).text();
        $(ex_data).empty();
    }

    pbl.initial = $(object).text();
    $(object).empty();

    pbl.solve = function() {
        if (pbl.running) {

        } else {
            var task = pbl.task;

            var url = problog.hostname + task.id;
            var data = task.collectData(pbl);

            if (data) {

                // Start task
                pbl.running = true;
                button_text = pbl.dom.edit_solve_btn.html();
                pbl.dom.edit_solve_btn
                    .prepend(' ')
                    .prepend($('<span>').addClass("glyphicon glyphicon-refresh glyphicon-refresh-animate"));
                pbl.dom.edit_solve_grp.removeClass('btn-group');
                pbl.dom.edit_solve_dwn.hide();

                $.ajax({
                      url: url,
                      dataType: 'jsonp',
                      data: data,
                      success:
                      function(data) {
                        // Reset interface
                        pbl.running = false;
                        pbl.dom.edit_solve_btn.html(button_text);
                        if (task.choices && task.choices.length > 0) {
                            pbl.dom.edit_solve_grp.addClass('btn-group');
                            pbl.dom.edit_solve_dwn.show();
                        }
                        console.log('result:', data);
                        if (data.SUCCESS == true) {
                            task.formatResult(pbl, data);
                            pbl.editor.getSession().clearAnnotations();
                            if (pbl.advanced) {
                                pbl.setSolveChoices(task.choices);
                            } else {
                                pbl.setSolveChoices();
                            }
                        } else {
                            p = data.err;
                            var msg = p.message;
                            if (msg == undefined) msg = p;
                            if (p.location && !p.location[0]) {
                                var row = p.location[1];
                                var col = p.location[2];
                            }
                            var result = $('<div>', { 'class': 'alert alert-danger' } ).text(msg);
                            if (row !== undefined) {
                                pbl.editor.getSession().setAnnotations([{ row: row-1, column: col, text: p.message, type: 'error'}]);
                            }
                            pbl.dom.results.html(result);
                        }
                        var cur_url = problog.main_editor_url + '#' + data.url;
                        if (pbl.trackurl) {
                            window.history.pushState({}, '', window.location.pathname + '#' + data.url);
                            cur_url = window.location.origin + window.location.pathname + '#' + data.url;
                        }

                        var clipboardBtn = $('<a class="problog-editor-hash" href="'+cur_url+'">Link to model&nbsp;<span class="problog-editor-hash glyphicon glyphicon-share" aria-hidden="true"></span></a>').appendTo(pbl.dom.results);
                        clipboardBtn.attr('title','Click to copy url to clipboard.');
                        clipboardBtn.click(function(e) {
                            e.preventDefault();
                            window.prompt("Copy to clipboard: Ctrl/Cmd+C, Enter", cur_url);
                        });

                      },
                      error:
                      function(data) {
                        // Reset interface
                        pbl.running = false;
                        pbl.dom.edit_solve_btn.html(button_text);
                        if (task.choices && task.choices.length > 0) {
                            pbl.dom.edit_solve_grp.addClass('btn-group');
                            pbl.dom.edit_solve_dwn.show();
                        }

                        // Show error message
                        var msg = 'The server returned an unexpected error.'
                        var result = $('<div>', { 'class': 'alert alert-danger' } ).text(msg);
                        pbl.dom.results.html(result);

                      }
                });
            } else {
                var msg = 'Please specify a model.';
                var result = $('<div>', { 'class': 'alert alert-danger' } ).text(msg);
                pbl.dom.results.html(result);
            }
        }
    };

    pbl.selectTaskByName = function(task) {
        $(problog.tasks).each(function(i,t) {
            if (task == t.id) {
                pbl.selectTask(i);
            }
        });
    }

    pbl.selectTask = function(taskid) {
        var task = problog.tasks[taskid];
        // Set task name on dropdown.
        pbl.dom.task_select_btn.text(task.name).append($("<span>").addClass("caret"));

        if (task.action === undefined) {
            pbl.dom.edit_solve_btn.text('Evaluate');
        } else {
            pbl.dom.edit_solve_btn.text(task.action);
        }

        if (pbl.task !== undefined) {
            pbl.task.deselect(pbl);
        }
        task.select(pbl);
        pbl.task = task;
        pbl.setSolveChoices(task.choices);

    };

    pbl.dom.root = $(object);

    pbl.trackurl = pbl.dom.root.data('trackurl');

    var task = pbl.dom.root.data('task');
    pbl.showtasks = (task == 'all');
    pbl.advanced = (pbl.dom.root.data('advanced') == true);
    pbl.taskid = 0;
    $(problog.tasks).each(function(i,t) {
        if (task == t.id) {
            pbl.taskid = i;
        }
    });

    // Components involved:
    //  - taskpane: Task selection (optional)
    //      - taskselect (group, button, list)
    //  - editpane: Editor and solving
    //      - editor
    //      - options
    //      - solve
    //  - resultpane: Results

    pbl.setSolveChoices = function(choices) {
        pbl.dom.edit_solve_lst.empty();
        if (choices && choices.length > 0) {
            pbl.dom.edit_solve_grp.addClass("btn-group");
            pbl.dom.edit_solve_dwn.show();
            $(choices).each(function(i,c) {
                var cname = c.name;
                if (cname == '-') {
                    pbl.dom.edit_solve_lst.append(
                        $('<li>').attr('role','separator').addClass('divider'));
                } else if (cname.substr(0,1) == '-') {
                    pbl.dom.edit_solve_lst.append(
                        $('<li>').addClass('dropdown-header').text(cname.substr(1)));
                } else {
                    pbl.dom.edit_solve_lst.append(
                        $('<li>').append($('<a>').attr('href','#')
                                                 .text(c.name)
                                                 .click(function() {
                                                    pbl.dom.edit_solve_btn.text(pbl.task.action + ' (' + pbl.task.choices[i].name + ')');
                                                    pbl.solve_choice = i;
                                                 })));
                }
            });
        } else {
            pbl.dom.edit_solve_grp.removeClass("btn-group");
            pbl.dom.edit_solve_dwn.hide();
            pbl.solve_choice = 0;
        }
    };

    pbl.dom.taskpane = $('<div>').addClass('problog-taskpane col-md-2 col-xs-12 pull-right')
                                 .appendTo(object);


    pbl.dom.task_select_grp = $('<div>').addClass("btn-group")
                                        .attr("style","width: 100%")
                                        .appendTo(pbl.dom.taskpane);

    pbl.dom.task_select_btn = $("<button>").addClass("btn btn-default dropdown-toggle")
                                           .attr("style","width: 100%")
                                           .attr("data-toggle","dropdown")
                                           .attr("aria-haspopup","true")
                                           .attr("aria-expanded","false")
                                           .appendTo(pbl.dom.task_select_grp);

    pbl.dom.task_select_lst = $("<ul>").addClass("dropdown-menu")
                                       .attr("style", "width: 100%")
                                       .appendTo(pbl.dom.task_select_grp);

    $(problog.tasks).each(function(i,t) {
        pbl.dom.task_select_lst.append(
            $("<li>").append($("<a>").attr('href','#')
                                     .text(t.name)
                                     .click(function() {
                                         pbl.selectTask(i);
                                     })
                            )
        );
    });

    if (pbl.showtasks) {
        pbl.dom.taskpane.show();
    } else {
        pbl.dom.taskpane.hide();
    }

    pbl.dom.editpane = $('<div>').addClass('problog-editpane').appendTo(object);

    pbl.dom.root.preface = $('<div>').appendTo(pbl.dom.editpane);

    if (pbl.showtasks) {
        pbl.dom.editpane.addClass('col-md-10 col-xs-12');
    } else {
        pbl.dom.editpane.addClass('col-md-12 col-xs-12');
    }

    // Initialize editor
    pbl.dom.edit_editor = $('<div>').addClass("problog-edit-editor").appendTo(pbl.dom.editpane);
    pbl.editor = ace.edit(pbl.dom.edit_editor[0]);
    pbl.editor.getSession().setMode('ace/mode/problog');
    pbl.editor.getSession().setUseWrapMode(true);
    pbl.editor.setShowInvisibles(false);
    pbl.editor.setValue(pbl.initial, -1);
    if (pbl.dom.root.data('autosize')) {
        pbl.editor.setOptions({
            maxLines: Infinity
        });
    }

    // Initialize edit options
    pbl.dom.edit_options = $('<div>').addClass("problog-edit-options").appendTo(pbl.dom.editpane);

    // Intialize solve pane
    pbl.dom.edit_solve = $('<div>').addClass("problog-edit-solve").appendTo(pbl.dom.editpane);

    // Initialize solve options
    pbl.dom.edit_solve_options = $('<div>').addClass("problog-edit-solve-options");

    // Initialize solve button (group)
    pbl.dom.edit_solve_grp = $('<div>').addClass("btn-group pull-right")
                                       .appendTo(pbl.dom.edit_solve);
    pbl.dom.edit_solve_btn = $('<button>').addClass("btn btn-primary")
                                          .text('Evaluate')
                                          .click(function() { pbl.solve() })
                                          .appendTo(pbl.dom.edit_solve_grp);
    pbl.dom.edit_solve_dwn = $("<button>").addClass("btn btn-primary dropdown-toggle")
                                          .attr("data-toggle","dropdown")
                                          .attr("aria-haspopup","true")
                                          .attr("aria-expanded","false")
                                          .append($("<span>").addClass("caret"))
                                          .appendTo(pbl.dom.edit_solve_grp);
    pbl.dom.edit_solve_lst = $("<ul>").addClass("dropdown-menu")
                                      .appendTo(pbl.dom.edit_solve_grp);

    // Initialize result pane
    pbl.dom.resultpane = $('<div>').addClass('problog-resultpane col-md-12 col-xs-12 panel panel-default').appendTo(object);

    pbl.dom.results = $('<div>').addClass('panel-body').appendTo(pbl.dom.resultpane);

    pbl.selectTask(pbl.taskid);
    if (pbl.advanced) {
        pbl.setSolveChoices(pbl.task.choices);
    } else {
        pbl.setSolveChoices();
    }

    if (pbl.trackurl) {
        problog.fetchModel(pbl);
        problog.trackUrlHash(pbl);
    }
    problog.editors.push(pbl);

}


problog.trackUrlHash = function(pbl) {
//  $(window).on('hashchange', function() {
//    problog.fetchModel(pbl);
//  });
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
    var a = $(a).children('td').eq(col).text().toLowerCase();
    var b = $(b).children('td').eq(col).text().toLowerCase();
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

problog.createTable = function(body, columns) {

    // Create table head

    head = $('<tr>');
    $(columns).each(function(index, elem) {
        head.append($('<th class="problog-result-sortable">').css('width', elem[1]).text(elem[0]));
    });

    result = $('<table>').addClass('table table-condensed')
                         .append($('<thead>').append(head))
                         .append(body);

    var table = result[0];
    problog.sortTable(table, 1, 0);

    // Create sortable table
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

    return table;
}


/** Load model from hash in editor **/
problog.fetchModel = function(pbl) {
    var default_task = pbl.dom.root.data('task');
    var task = problog.getTaskFromUrl(default_task);

    pbl.selectTaskByName(task);

    var hash = problog.getHashFromUrl();
    if (hash) {
        $.ajax({
            url: problog.hostname+'model',
            dataType: 'jsonp',
            data: {'hash': hash},
        }).done( function(data) {
            if (data.SUCCESS == true) {
                pbl.editor.setValue(data.model,-1);
            } else {
                pbl.editor.setValue('% '+data.err,-1);
            }
        }).fail( function(jqXHR, textStatus, errorThrown) {
            pbl.editor.setValue(jqXHR.responseText);
        });
    }

    var ehash = problog.getExamplesHashFromUrl();
    if (ehash && pbl.editor_data) {
        $.ajax({
            url: problog.hostname+'examples',
            dataType: 'jsonp',
            data: {'hash': ehash},
        }).done( function(data) {
            console.log(data);
            if (data.SUCCESS == true) {
                pbl.editor_data.setValue(data.examples,-1);
            } else {
                pbl.editor_data.setValue('% '+data.err,-1);
            }
        }).fail( function(jqXHR, textStatus, errorThrown) {
            pbl.editor_data.setValue(jqXHR.responseText);
        });
    }

    // window.history.pushState({}, '', window.location.pathname);

};



problog.getTaskFromUrl = function(default_task) {
  var hash = window.location.hash;
  hashidx = hash.indexOf("task=");
  if (hashidx > 0) {
    hash = hash.substr(hashidx+5, 32);
    ampidx = hash.indexOf("&");
    if (ampidx > 0) {
      hash = hash.substring(0, ampidx);
    }
  } else {
    if (problog.getExamplesHashFromUrl()) {
        hash = 'lfi';
    } else {
        hash = default_task;
    }
  }
  return hash;
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
