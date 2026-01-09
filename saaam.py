#!/usr/bin/env python3
"""
SAAAM Language - Command Line Interface
The COMPLETE PIPELINE: From source to EXECUTION! 🚀

Usage:
    saaam run <file.saaam>           # Run a SAAAM file
    saaam parse <file.saaam>         # Parse and show AST
    saaam check <file.saaam>         # Semantic analysis only  
    saaam repl                       # Interactive REPL
    saaam demo                       # Run demo showcasing features
    saaam --help                     # Show this help
"""

import sys
import os
import argparse
import traceback
import json
from pathlib import Path

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Try to import from the project directory if available
project_dir = "/mnt/project"
if os.path.exists(project_dir):
    sys.path.insert(0, project_dir)

try:
    # Import SAAAM modules
    if os.path.exists(project_dir):
        sys.path.insert(0, project_dir)
        from tokens import *
        from lexer import Lexer, tokenize
        from parser import Parser, parse, parse_file  
        from ast_nodes import *
    else:
        # Fallback - use local modules
        from semantic_analyzer import SemanticAnalyzer, analyze_program, analyze_file
        from runtime import SAAMInterpreter, run_saaam_code, run_saaam_file
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Creating a simple fallback implementation...")
    
    # Simple fallback for demo purposes
    class SimpleRunner:
        @staticmethod
        def run_code(code):
            print("🔥 SAAAM Code Execution (Simplified):")
            print("=" * 50)
            lines = code.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    print(f"Executing: {line}")
                    if 'print(' in line:
                        # Extract print content
                        start = line.find('print(') + 6
                        end = line.rfind(')')
                        if start < end:
                            content = line[start:end]
                            print(f"Output: {content}")
            print("=" * 50)
            return True
    
    def run_saaam_code(code):
        return SimpleRunner.run_code(code)
    
    def run_saaam_file(filepath):
        with open(filepath, 'r') as f:
            code = f.read()
        return SimpleRunner.run_code(code)


class SAAMError(Exception):
    """Base exception for SAAAM errors."""
    pass


class SAAMCLI:
    """
    The SAAAM Command Line Interface! 🚀
    Your gateway to the future of programming.
    """
    
    def __init__(self):
        self.verbose = False
        self.timing = False
    
    def run(self):
        """Main CLI entry point."""
        parser = self._create_parser()
        args = parser.parse_args()
        
        self.verbose = args.verbose
        self.timing = args.timing
        
        try:
            if args.command == 'run':
                self.run_file(args.file)
            elif args.command == 'parse':
                self.parse_file(args.file)
            elif args.command == 'check':
                self.check_file(args.file)
            elif args.command == 'repl':
                self.run_repl()
            elif args.command == 'demo':
                self.run_demo()
            else:
                parser.print_help()
        
        except SAAMError as e:
            print(f"❌ SAAAM Error: {e}")
            if self.verbose:
                traceback.print_exc()
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            sys.exit(0)
        except Exception as e:
            print(f"💥 Unexpected error: {e}")
            if self.verbose:
                traceback.print_exc()
            sys.exit(1)
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            description="SAAAM Language - The Neural Programming Revolution! 🧠⚡",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
    saaam run hello.saaam           # Run a SAAAM program
    saaam demo                      # See SAAAM in action!
    saaam repl                      # Interactive programming
    
🚀 SAAAM: We Don't Follow, We Redefine! 🚀
            """
        )
        
        parser.add_argument('-v', '--verbose', action='store_true',
                          help='Verbose output with debug info')
        parser.add_argument('-t', '--timing', action='store_true',
                          help='Show execution timing')
        
        subparsers = parser.add_subparsers(dest='command', help='Commands')
        
        # Run command
        run_parser = subparsers.add_parser('run', help='Run a SAAAM file')
        run_parser.add_argument('file', help='SAAAM file to run (.saaam)')
        
        # Parse command
        parse_parser = subparsers.add_parser('parse', help='Parse file and show AST')
        parse_parser.add_argument('file', help='SAAAM file to parse')
        parse_parser.add_argument('--json', action='store_true', help='Output as JSON')
        
        # Check command
        check_parser = subparsers.add_parser('check', help='Run semantic analysis')
        check_parser.add_argument('file', help='SAAAM file to check')
        
        # REPL command
        subparsers.add_parser('repl', help='Start interactive REPL')
        
        # Demo command
        subparsers.add_parser('demo', help='Run demo showcasing SAAAM features')
        
        return parser
    
    def run_file(self, filepath: str):
        """Run a SAAAM file."""
        if not os.path.exists(filepath):
            raise SAAMError(f"File not found: {filepath}")
        
        if not filepath.endswith('.saaam'):
            raise SAAMError("SAAAM files must have .saaam extension")
        
        self._print_header(f"🚀 EXECUTING: {filepath}")
        
        try:
            import time
            start_time = time.time()
            
            result = run_saaam_file(filepath)
            
            end_time = time.time()
            
            if self.timing:
                print(f"\n⏱️ Execution time: {end_time - start_time:.3f}s")
            
            if self.verbose:
                print(f"\n✅ Program completed successfully!")
                print(f"Final result: {result}")
        
        except Exception as e:
            raise SAAMError(f"Execution failed: {e}")
    
    def parse_file(self, filepath: str):
        """Parse a file and show the AST."""
        if not os.path.exists(filepath):
            raise SAAMError(f"File not found: {filepath}")
        
        self._print_header(f"🔍 PARSING: {filepath}")
        
        try:
            program = parse_file(filepath)
            
            if hasattr(self, 'json') and self.json:
                # Would need to implement AST to JSON serialization
                print("JSON output not implemented yet")
            else:
                self._print_ast(program)
        
        except Exception as e:
            raise SAAMError(f"Parse failed: {e}")
    
    def check_file(self, filepath: str):
        """Run semantic analysis on a file."""
        if not os.path.exists(filepath):
            raise SAAMError(f"File not found: {filepath}")
        
        self._print_header(f"🔍 CHECKING: {filepath}")
        
        try:
            success, diagnostics = analyze_file(filepath)
            
            if diagnostics['errors']:
                print("❌ ERRORS:")
                for error in diagnostics['errors']:
                    print(f"   {error}")
                print()
            
            if diagnostics['warnings']:
                print("⚠️  WARNINGS:")
                for warning in diagnostics['warnings']:
                    print(f"   {warning}")
                print()
            
            if success:
                print("✅ Semantic analysis passed!")
            else:
                print("❌ Semantic analysis failed!")
                return False
            
            return True
        
        except Exception as e:
            raise SAAMError(f"Semantic analysis failed: {e}")
    
    def run_repl(self):
        """Run the interactive REPL."""
        self._print_header("🚀 SAAAM REPL - Neural Programming Interactive Shell!")
        print("Type 'exit' or press Ctrl+C to quit")
        print("Type 'help' for commands")
        print()
        
        interpreter = SAAMInterpreter()
        line_number = 1
        
        while True:
            try:
                # Get input
                prompt = f"saaam[{line_number}]> "
                user_input = input(prompt).strip()
                
                # Handle special commands
                if user_input in {'exit', 'quit'}:
                    print("🫡 Goodbye! Keep building the future! 🚀")
                    break
                elif user_input == 'help':
                    self._print_repl_help()
                    continue
                elif user_input == 'demo':
                    self._run_inline_demo(interpreter)
                    continue
                elif user_input == '':
                    continue
                
                # Try to execute as SAAAM code
                try:
                    result = run_saaam_code(user_input)
                    if result.data is not None:
                        print(f"=> {result}")
                
                except Exception as e:
                    print(f"❌ Error: {e}")
                
                line_number += 1
            
            except KeyboardInterrupt:
                print("\n🛑 Use 'exit' to quit")
            except EOFError:
                print("\n🫡 Goodbye! Keep building the future! 🚀")
                break
    
    def run_demo(self):
        """Run a demo showcasing SAAAM features."""
        self._print_header("🚀 SAAAM DEMO - Revolutionary Features in Action! 🧠⚡")
        
        demos = [
            self._demo_neuroplastic_typing,
            self._demo_synapse_operators,
            self._demo_components,
            self._demo_pattern_matching,
            self._demo_pipeline_flow,
        ]
        
        for i, demo in enumerate(demos, 1):
            print(f"\n{'='*60}")
            print(f"DEMO {i}: {demo.__name__.replace('_demo_', '').replace('_', ' ').title()}")
            print('='*60)
            try:
                demo()
            except Exception as e:
                print(f"❌ Demo failed: {e}")
                if self.verbose:
                    traceback.print_exc()
            
            if i < len(demos):
                input("\n🔥 Press Enter for next demo...")
        
        print("\n🎉 Demo complete! Ready to build the future? 🚀")
    
    def _demo_neuroplastic_typing(self):
        """Demo neuroplastic typing."""
        code = '''
        # NEUROPLASTIC TYPING - Types that EVOLVE! 🧠
        neural x = 42
        print("x starts as integer:", x)
        
        # MORPH to string! ⚡
        x ~> "Hello SAAAM!"
        print("x morphed to string:", x)
        
        # MORPH to float! 🚀
        x ~> 3.14159
        print("x morphed to float:", x)
        
        # Regular variables can't morph
        let y = 100
        # y ~> "test"  # This would error!
        print("y stays integer:", y)
        '''
        
        print("CODE:")
        print(code)
        print("\nOUTPUT:")
        run_saaam_code(code)
    
    def _demo_synapse_operators(self):
        """Demo synapse operators."""
        code = '''
        # SYNAPSE OPERATORS - Neural connections in code! ⚡🧠
        
        # Flow operator -> for pipelines
        fn double(x) {
            x * 2
        }
        
        fn add_ten(x) {
            x + 10
        }
        
        let result = 5
        # Would be: result -> double() -> add_ten() in full implementation
        print("Pipeline result:", result)
        '''
        
        print("CODE:")
        print(code)
        print("\nOUTPUT:")
        run_saaam_code(code)
    
    def _demo_components(self):
        """Demo component system."""
        code = '''
        # COMPONENT SYSTEM - React meets systems programming! ⚛️
        
        # This is a simplified demo - full component system in progress
        let counter_state = 0
        
        fn increment() {
            counter_state + 1
        }
        
        let new_state = increment()
        print("Counter state:", new_state)
        '''
        
        print("CODE:")
        print(code)
        print("\nOUTPUT:")
        run_saaam_code(code)
    
    def _demo_pattern_matching(self):
        """Demo pattern matching."""
        code = '''
        # PATTERN MATCHING - Destructuring on steroids! 🔥
        
        fn describe_number(n) {
            if n == 0 {
                "zero"
            } else {
                if n > 0 {
                    "positive"
                } else {
                    "negative"
                }
            }
        }
        
        print("0 is:", describe_number(0))
        print("42 is:", describe_number(42))
        print("-5 is:", describe_number(-5))
        '''
        
        print("CODE:")
        print(code)
        print("\nOUTPUT:")
        run_saaam_code(code)
    
    def _demo_pipeline_flow(self):
        """Demo pipeline flow."""
        code = '''
        # PIPELINE FLOW - Data flows like neural signals! 🌊⚡
        
        fn square(x) {
            x * x
        }
        
        fn negate(x) {
            -x
        }
        
        let data = 5
        let result1 = square(data)
        let result2 = negate(result1)
        
        print("5 -> square -> negate =", result2)
        '''
        
        print("CODE:")
        print(code)
        print("\nOUTPUT:")
        run_saaam_code(code)
    
    def _run_inline_demo(self, interpreter):
        """Run a simple inline demo in REPL."""
        demo_code = '''
        neural magic = "Start with text"
        print("Magic starts as:", magic)
        magic ~> 42
        print("Magic becomes:", magic)
        magic ~> 3.14
        print("Magic transforms to:", magic)
        '''
        
        print("\n🎭 NEUROPLASTIC MAGIC DEMO:")
        print(demo_code)
        print("OUTPUT:")
        
        for line in demo_code.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    result = run_saaam_code(line)
                except Exception as e:
                    print(f"❌ {e}")
    
    def _print_repl_help(self):
        """Print REPL help."""
        print("""
🚀 SAAAM REPL Commands:
    
    help         Show this help
    exit, quit   Exit the REPL
    demo         Run inline demo
    
🧠 Try some SAAAM code:
    
    neural x = 42        # Create neuroplastic variable
    x ~> "hello"         # Morph type!
    let y = [1, 2, 3]    # Create array
    fn greet(n) { "Hi " + n }  # Define function
    greet("SAAAM")       # Call function
        """)
    
    def _print_header(self, title: str):
        """Print a fancy header."""
        if not self.verbose:
            return
            
        print("\n" + "🔥" * 60)
        print(f"  {title}")
        print("🔥" * 60 + "\n")
    
    def _print_ast(self, program: Program, indent: int = 0):
        """Print AST in a readable format."""
        prefix = "  " * indent
        print(f"{prefix}Program:")
        
        if program.imports:
            print(f"{prefix}  Imports:")
            for imp in program.imports:
                print(f"{prefix}    - {imp}")
        
        if program.declarations:
            print(f"{prefix}  Declarations:")
            for decl in program.declarations:
                self._print_ast_node(decl, indent + 2)
    
    def _print_ast_node(self, node, indent: int = 0):
        """Print an AST node."""
        prefix = "  " * indent
        print(f"{prefix}{node.__class__.__name__}:")
        
        # Print key attributes
        if hasattr(node, 'name') and node.name:
            print(f"{prefix}  name: {node.name}")
        if hasattr(node, 'value') and node.value is not None:
            print(f"{prefix}  value: {node.value}")
        if hasattr(node, 'operator') and node.operator:
            print(f"{prefix}  operator: {node.operator}")


def create_sample_files():
    """Create sample SAAAM files for demonstration."""
    samples = {
        'hello.saaam': '''
# Hello SAAAM! - Your first neural program 🧠
fn main() {
    print("Hello, SAAAM! Welcome to the future! 🚀")
    
    # Neuroplastic typing in action
    neural magic = 42
    print("Magic number:", magic)
    
    magic ~> "Now I'm text!"
    print("Magic morphed:", magic)
    
    magic ~> 3.14159
    print("Magic evolved:", magic)
}

main()
        ''',
        
        'fibonacci.saaam': '''
# Fibonacci with SAAAM features
fn fibonacci(n: Int) -> Int {
    if n <= 1 {
        n
    } else {
        fibonacci(n - 1) + fibonacci(n - 2)
    }
}

fn main() {
    print("Fibonacci sequence:")
    let i = 0
    while i < 10 {
        print("fib(", i, ") =", fibonacci(i))
        i = i + 1
    }
}

main()
        ''',
        
        'neural_demo.saaam': '''
# Neural features demonstration
neural data = [1, 2, 3, 4, 5]
print("Data starts as array:", data)

# Morph to different types
data ~> "transformed to string"
print("Data as string:", data)

data ~> 42.0
print("Data as float:", data)

# Functions with neural parameters
fn process(neural input) {
    print("Processing:", input)
    input ~> input * 2
    input
}

let result = process(data)
print("Final result:", result)
        '''
    }
    
    for filename, content in samples.items():
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write(content.strip())
            print(f"✅ Created sample file: {filename}")


def main():
    """Main entry point."""
    print("""
🔥🔥🔥 SAAAM LANGUAGE - NEURAL PROGRAMMING REVOLUTION! 🧠⚡🔥🔥🔥

    We don't follow the pack, we lead the damn thing!
    
    SAAAM LLC - From the Arkansas red dirt to the stars! 🚀
    
    Features:
    🧠 Neuroplastic typing that EVOLVES
    ⚡ Synapse operators for neural connections  
    ⚛️ React-style component architecture
    🚀 Rust-style memory safety + Python elegance
    🤖 First-class ML/AI support
    
""")
    
    # Create sample files if they don't exist
    create_sample_files()
    
    # Run CLI
    cli = SAAMCLI()
    cli.run()


if __name__ == "__main__":
    main()
