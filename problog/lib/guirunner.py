"""
Part of the ProbLog distribution.

Copyright 2015 KU Leuven, DTAI Research Group

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import print_function

import argparse
import sys
import subprocess
import os


if sys.version_info.major == 2 :
    from Tkinter import Tk, BOTH, RIGHT, IntVar, StringVar, Grid, X, Y,  Text, END, Spinbox
    from ttk import Combobox, Entry, Frame, Button, Label, Checkbutton
    import tkFileDialog
else :
    from tkinter import Tk, BOTH, RIGHT, IntVar, StringVar, Grid, X, Y,  Text, END, Spinbox
    from tkinter.ttk import Combobox, Entry, Frame, Button, Label, Checkbutton
    import tkinter.filedialog as tkFileDialog

from idlelib.WidgetRedirector import WidgetRedirector

class ReadOnlyText(Text) :

    def __init__(self, *args, **kwdargs) :
        Text.__init__(self, *args, **kwdargs)
        self.redirector = WidgetRedirector(self)
        self.insert = self.redirector.register("insert", lambda *args,**kw: "break")
        self.delete = self.redirector.register("delete", lambda *args,**kw: "break")
        self.replace = self.redirector.register("replace", lambda *args,**kw: "break")

    def setText(self, text) :
        self.delete(1.0, END)
        self.insert(END, text)


    # def insert(self, *args, **kwdargs) :
    #
    #     print ('INSERTING')
    #
    #     return "break"
    #
    # def delete(self, *args, **kwdargs) :
    #     return "break"
    #
    # def replace(self, *args, **kwdargs) :
    #     return "break"



class MainWindow(Frame) :

    def __init__(self, parent, parser, scriptname, progname=None) :
        Frame.__init__(self, parent)
        self.scriptname = scriptname
        self.progname = progname
        self.function = []
        self.variables = []
        self.parent = parent
        self.initUI(parser)

    def centerWindow(self) :
        w = 500
        h = 300

        sw = self.parent.winfo_screenwidth()
        sh = self.parent.winfo_screenheight()

        x = (sw - w)/2
        y = (sh - h)/2
        self.parent.geometry("%dx%d+%d+%d" % (w,h,x,y))

    def initUI(self, parser) :
        # TODO required arguments
        # TODO repeated arguments (e.g. filenames)

        self.parent.title(self.progname)
        self.pack(fill=BOTH, expand=1)
        self.centerWindow()

        Grid.rowconfigure(self,1,weight=1)
        Grid.columnconfigure(self,0,weight=1)

        inputFrame = Frame(self)
        inputFrame.grid(row=0, column=0, sticky='WE')

        outputFrame = Frame(self)
        outputFrame.grid(row=1, column=0, sticky='WENS')

        self.outputText = ReadOnlyText(outputFrame)
        self.outputText.pack(fill=BOTH, expand=1)

        # Main controls frame
        mainFrame = Frame(inputFrame)
        mainFrame.pack(fill=BOTH, expand=1)

        # Auto-resize column
        Grid.columnconfigure(mainFrame,1,weight=1)

        # Ok button
        okButton = Button(inputFrame, text='Run', command=self.run, default='active')
        okButton.pack(side=RIGHT, padx=5, pady=5)

        # Cancel button
        cancelButton = Button(inputFrame, text='Exit', command=self.quit)
        cancelButton.pack(side=RIGHT)

        # Add controls to mainframe for all options
        for index, action in enumerate(parser._actions) :
            action_type = type(action).__name__
            if action_type == '_HelpAction' :
                pass
            else :
                self.function.append(lambda v : [v])
                self.variables.append(None)
                if action.choices :
                    self._add_choice( mainFrame, index, action.dest, action.choices, action.default, action.option_strings[0] )
                elif action_type == '_StoreTrueAction' :
                    self._add_check( mainFrame, index, action.dest, False, action.option_strings[0] )
                elif action_type == '_CountAction' :
                    self._add_count( mainFrame, index, action.dest, 0, action.option_strings[0] )
                elif action.type and action.type.__name__ == 'inputfile' :
                    self._add_filename( mainFrame, index, action.dest, 'r', action.option_strings )
                elif action.type and action.type.__name__ == 'outputfile' :
                    self._add_filename( mainFrame, index, action.dest, 'w', action.option_strings )
                else :
                    self._add_field( mainFrame, index, action.dest )

    def _add_field( self, frame, index, name ) :
        self.variables[-1] = StringVar()
        label = Label(frame, text=name)
        label.grid(row=index, column=0, sticky='W', padx=10)
        field = Entry(frame)
        field.grid(row=index, column=1, sticky='WE', textvariable=self.variables[-1])

    def _add_count(self, frame, index, name, default, option) :
        self.function[-1] = lambda v : [option]*int(v)
        self.variables[-1] = StringVar()
        self.variables[-1].set(default)
        label = Label(frame, text=name)
        label.grid(row=index, column=0, sticky='W', padx=10)
        field = Spinbox(frame, from_=0, to=100, textvariable=self.variables[-1])
        field.grid(row=index, column=1, sticky='WE')


    def _add_choice(self, frame, index, name, choices, default, option) :
        self.function[-1] = lambda v : [ option, v ]
        label = Label(frame, text=name)
        label.grid(row=index, column=0, sticky='W', padx=10)
        field = Combobox(frame, values=choices, state='readonly' )
        field.set(default)
        field.grid(row=index, column=1, sticky='WE')
        self.variables[-1] = field

    def _add_check(self, frame, index, name, default, option) :
        self.variables[-1] = StringVar()
        self.variables[-1].set('')

        label = Label(frame, text=name)
        label.grid(row=index, column=0, sticky='W', padx=10)
        field = Checkbutton(frame, variable=self.variables[-1], onvalue=option, offvalue='')
        field.grid(row=index, column=1, sticky='WE')

    def _add_filename(self, frame, index, name, mode, option) :
        if option :
            self.function[-1] = lambda v : [option, v]
        else :
            self.function[-1] = lambda v : [v]

        self.variables[-1] = StringVar()
        var = self.variables[-1]

        def set_name() :
            if mode == 'r' :
                fn = tkFileDialog.askopenfilename(initialdir='.')
            else :
                fn = tkFileDialog.asksaveasfilename(initialdir='.')
            var.set(fn)

        label = Label(frame, text=name)
        label.grid(row=index, column=0, sticky='W', padx=10)

        field_button = Frame(frame)
        Grid.columnconfigure(field_button,0,weight=1)

        field = Entry(field_button, textvariable=var)
        field.grid(row=0, column=0, sticky='WE')
        button = Button(field_button, text="...", command=set_name, width=1, padding=0)
        button.grid(row=0,column=1)

        field_button.grid(row=index, column=1, sticky='WE')

    def run(self) :

        # TODO enforce required arguments
        # TODO show errors

        args = []
        for i, x in enumerate(self.variables) :
            if self.function[i] != None and x.get() :
                args += self.function[i](x.get())

        cmd = [sys.executable, self.scriptname] + args

        result = (subprocess.check_output(cmd))

        self.outputText.setText(result)

def show_gui( scriptname, progname=None ) :
    modulename = os.path.splitext(os.path.basename(scriptname))[0]
    script = __import__(modulename)
    parser = script.argparser()

    if progname is None : progname = os.path.basename(scriptname)

    root = Tk()
    app = MainWindow(root, parser, scriptname, progname)
    root.mainloop()

if __name__ == '__main__' :
    scriptname = sys.argv[1]
    if len(sys.argv) > 2 :
        progname = sys.argv[2]
    else :
        progrname = None

    show_gui(scriptname, progname)
